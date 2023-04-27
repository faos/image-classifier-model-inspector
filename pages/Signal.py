import streamlit as st
import numpy as np
from PIL import Image
import copy

from streamlit_drawable_canvas import st_canvas

import plotly.express as px
import plotly.graph_objects as go

from image_processing import texture
from interpretability_utils import methods
from model_utils import inference

st.set_page_config(layout='wide')


def build_dataframe(logits):
    data = {
        "categories":list(range(len(logits))),
        "probabilities":logits
        }

    return data


def build_dataframe_pair(logits_raw, logits_wt):
    data = {
        "categories":list(range(len(logits_raw))) + list(range(len(logits_wt))),
        "probabilities":logits_raw + logits_wt,
        "type":['Original'] * len(logits_raw) + ['Signal'] * len(logits_raw)
        }
    
    return data


def build_dataframe_pair_background(logits_raw, logits_wt):
    data = {
        "categories": list(range(len(logits_raw))) + list(range(len(logits_wt))),
        "probabilities": logits_raw + logits_wt,
        "type": ['Original'] * len(logits_raw) + ['Back.'] * len(logits_raw)
    }

    return data


if "shared" not in st.session_state:
   st.session_state["shared"] = True

if st.session_state.get('input-image-file', False):
    input_image = st.session_state['input-image-file']
    st.sidebar.image(input_image, use_column_width=True)
    bg_image = copy.deepcopy(input_image)
    set_img_size = st.sidebar.select_slider(
        label="Select image size",
        value=st.session_state['img_size'],
        options=[32, 96, 224, 299]
    )
    set_mean = st.sidebar.text_input(label="Mean", value=st.session_state['set_mean'])
    set_std = st.sidebar.text_input(label="Std", value=st.session_state['set_std'])
    #TO DO: Validate mean and std format
    st.session_state['set_mean'] = eval(set_mean)
    st.session_state['set_std'] = eval(set_std)
else:
    bg_image = None

img_size = 224

interpretability_method = 'saliency'

stroke_width = 1
stroke_color = "#ffffff"

bg_color = "#000000"
img_size = set_img_size

st.header('Select the signal region from the input')
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)
col_cfg = (.1, 5.75, .05, 5.75, .1)
row0_spc1, row0_2, row0_spc3, row0_3, row0_spc4 = st.columns(col_cfg)

with row0_2:
    # Specify canvas parameters in application
    drawing_mode = st.selectbox(
        "Drawing tool:", ("rect", "circle", "polygon")
    )

with row0_3:
    # Specify canvas parameters in application
    draw_operation = "Selection"
    draw_operation = st.selectbox(
        "Drawing operation:", ("Selection", "Erase")
    )

if bg_image:
    target = 0
    img = Image.open(bg_image).resize((img_size, img_size))
    att = methods.Interpreter(
        st.session_state["model"],
        interpretability_method,
        size=img_size
    )(img, target)

    draw_space1, draw_col1, draw_col2, draw_col3, draw_space3 = st.columns((.1, 2.0, 2.0, 2.0, .1))

    with draw_col1:
        st.write("")

    with draw_col2:
        realtime_update = False

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=img,
            update_streamlit=realtime_update,
            height=img.size[1],
            width=img.size[0],
            drawing_mode=drawing_mode,
            point_display_radius=0,
            display_toolbar=True,
            key="canvas",
        )

    with draw_col3:
        st.write("")

st.header('Signal-to-noise Analysis')
st.markdown("""The signal-to-noise analysis computes the model inference using only the signal and compares it with the original image inference. Besides, it presents the signal importance vs. background importance using the interpretability of the original image.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)
if bg_image:

    img = Image.open(bg_image).resize((img_size, img_size))
    vec = np.asarray(img)
    height, width, channels = vec.shape
    row01_spacer1, row01_2, row01_spacer3, row01_3, row01_spacer4 = st.columns((.1, 2.0, .1, 2.0, .1))

    with row01_2:

        if canvas_result.image_data is not None:
            mask = Image.fromarray(canvas_result.image_data[:,:,1:])
            mask = mask.resize((width, height))
            mask = np.asarray(mask)

            print('Stats:')
            print("\tSUMMARY", type(canvas_result.image_data[:, :, 1:]), canvas_result.image_data.shape)
            print("\tMASK SUM:", (mask > 0).sum())
            print("\tERASE MASK SUM:", (1 - (mask > 0)).sum(), (1 - (mask > 0)).shape, ((1 - (mask > 0)) < 0).sum())
            print('\tMASK:', mask.shape)

            if draw_operation == 'Erase':
                img_ans = Image.fromarray((1 - (mask > 0)).astype(np.uint8) * vec, mode="RGB")
            
            elif draw_operation == 'Selection':
                img_ans = Image.fromarray((mask > 0) * vec, mode="RGB")

            st.image(img_ans.resize(img.size), use_column_width=True)

    with row01_3:
        if canvas_result.image_data is not None:
            signal_mask = ((mask > 0)).astype(np.uint8).mean(2)
            background_mask = (1 - (mask > 0)).astype(np.uint8).mean(2)

            signal = signal_mask * att
            background = background_mask * att
            print(signal.sum(), signal_mask.sum())
            print(background.sum(), background_mask.sum())
            heatmap = np.zeros(signal_mask.shape)

            heatmap[signal_mask != 0.0] = signal.sum() / signal_mask.sum()
            heatmap[background_mask != 0.0] = background.sum() / background_mask.sum()

            signal_weight = signal.sum() / signal_mask.sum()
            noise_weight = background.sum() / background_mask.sum()
            total_weight = signal_weight + noise_weight

            data_pie = {
                "weight":[signal_weight / total_weight, noise_weight / total_weight],
                "type":["Signal", "Noise"]
            }

            fig = go.Figure()

            fig.add_trace(go.Pie(labels=['Signal','Noise'],
                                 values=[signal_weight / total_weight,noise_weight / total_weight])
                          )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            #mask inference
            logits_raw = inference.Predictor()(st.session_state["model"], img)
            logits_txt = inference.Predictor()(st.session_state["model"], img_ans)

            df = build_dataframe_pair(logits_raw, logits_txt)
            fig = px.bar(
                df, x='categories', y='probabilities', color="type", barmode="group",
                title="Model inference on signal-only vs original image"
            )

            fig.update_layout(yaxis_range=[0, 1])
            fig.update_layout(
                font_size=25,
                legend_font_size=20,
                title_font_size=100,
            )
            fig['layout']['xaxis']['title'] = 'Category'
            fig['layout']['yaxis']['title'] = 'Probability (%)'
            fig['layout']['xaxis']['titlefont']['size'] = 25
            fig['layout']['yaxis']['titlefont']['size'] = 25

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.header('Texture Analysis')
st.markdown("""This analysis inserts noise on the image background, so the user can infer if the model changes its prediction when facing different background texture information but keeps the signal intact.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)

options = ["GaussianNoise", "ShotNoise", "ImpulseNoise", "SpekleNoise"]
col_cfg = (.1, 2.0, .1, 2.0, .1)
row0_spc2, row0_2, row0_spc3, row0_3, row0_spc4 = st.columns(col_cfg)

with row0_2:
    texture_option = st.selectbox(
        label="Select texture transforming",
        options=options,
        index=0
    )

with row0_3:
    texture_class = getattr(texture, texture_option)
    parameters = getattr(texture_class, "parameters")
    args = {}

    if parameters.get('scale', False):
        scale = st.slider('Scale', 0.0, 1.0, 0.5)
        args['scale'] = scale

    if parameters.get('lam_weight', False):
        lam_weight = st.slider('Lam weight', 0, 100, 1)
        args['lam_weight'] = lam_weight

    if parameters.get('amount', False):
        amount = st.slider('Amount', 0.0, 0.30, 0.03)
        args['amount'] = amount

if bg_image:
    col_cfg =(.1, 2, .1, 2, .1)
    row0_spc1, row0_1, row0_spc2, row0_2, row0_spc3 = st.columns(col_cfg)

    img = Image.open(bg_image).resize((img_size, img_size))
    vec = np.asarray(img)

    height, width, channels = vec.shape

    with row0_1:

        ans = texture_class(**args)(vec.copy())
        tmp = (1 - (mask>0)) * ans + (mask > 0) * vec
        tmp_int = (tmp).astype(np.uint8)
        background_texture_pilimg = Image.fromarray(tmp_int, mode="RGB")
        st.image(background_texture_pilimg, use_column_width=True)

    with row0_2:
        logits_raw = inference.Predictor()(
            st.session_state["model"],
            img
        )
        logits_txt = inference.Predictor()(
            st.session_state["model"],
            background_texture_pilimg
        )

        df = build_dataframe_pair_background(logits_raw, logits_txt)

        fig = px.bar(
            df, x='categories', y='probabilities',
            color="type", barmode="group",
            title="Model inference on background texture processing"
        )

        fig.update_layout(yaxis_range=[0, 1])
        fig.update_layout(
            font_size=25,
            legend_font_size=20,
            title_font_size=100,
        )
        fig['layout']['xaxis']['title'] = 'Category'
        fig['layout']['yaxis']['title'] = 'Probability (%)'
        fig['layout']['xaxis']['titlefont']['size'] = 25
        fig['layout']['yaxis']['titlefont']['size'] = 25

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


