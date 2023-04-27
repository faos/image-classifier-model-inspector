import streamlit as st
import copy
import numpy as np
from PIL import Image

from interpretability_utils import methods
from model_utils import inference
from interpretability_utils.u_analysis import analysis, get_u_positions

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(layout='wide')


def build_dataframe(logits):
    size_of_logits = len(logits)
    data = {
        "categories":list(range(size_of_logits)),
        "probabilities":logits
        }

    return data


def build_dataframe_pair(logits_raw, logits_wt):
    size_of_logits_raw = len(logits_raw)
    size_of_logits_wt = len(logits_wt)
    data = {
        "categories":list(range(size_of_logits_raw)) + list(range(size_of_logits_wt)),
        "probabilities":logits_raw + logits_wt,
        "type":['Raw'] * size_of_logits_raw + ['Texture'] * size_of_logits_wt
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
        options=[32, 96, 224, 299],
        key="interpretability-set-img-size"
    )
    set_mean = st.sidebar.text_input(label="Mean", value=st.session_state['set_mean'])
    set_std = st.sidebar.text_input(label="Std", value=st.session_state['set_std'])
    # TO DO: Validate mean and std format
    st.session_state['set_mean'] = eval(set_mean)
    st.session_state['set_std'] = eval(set_std)
else:
    bg_image = None

st.header('Interpretability')
st.markdown("""Performs the interpretability of the image classifier loaded. You should select the interpretability method and the model output target you want to interpret.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)


row0_spacer1, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .05, 2.0, 0.1))

with row0_2:
    # Specify canvas parameters in application
    interpretability_method = st.selectbox(
        "Method", ("saliency", "input-x-gradient", "integratedgradients", "deconvolution", "guided-backpropagation")
    )

with row0_3:
    # Specify canvas parameters in application
    target = st.slider(label="Target", max_value=st.session_state['num_classes'], min_value=0)

if bg_image:
    img = Image.open(bg_image).resize((set_img_size, set_img_size))
    row0_spacer0, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .05, 2.0, 0.1))

    with row0_2:
        att = methods.Interpreter(
            copy.deepcopy(st.session_state["model"]),
            interpretability_method
            )(img, target)
        att = (att - att.min())/(att.max() - att.min())

        fig = go.Figure()
        fig.add_trace(go.Image(z=img))
        fig.add_trace(
            go.Heatmap(
                z=att,
                zmin=0,
                zmax=1,
                colorscale="Jet",
                opacity=0.5
            )
        )
        fig.update_layout(
            title="Importance of each input pixel for model prediction", 
            coloraxis_showscale=False, 
            margin= dict(l=0,r=0,b=0),
            font=dict(size=18, color="Black")
            )
        fig.update_annotations(font_size=20)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with row0_3:
        logits_raw = inference.Predictor()(st.session_state["model"], img)
        df = build_dataframe(logits_raw)
        fig = px.bar(df, x='categories', y='probabilities')
        print("I N F E R E N C E - TEXTURE:", df)
        fig.update_layout(yaxis_range=[0, 1], margin= dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.header('U Analysis')
st.markdown(""" Run the U Analysis using the interpretability obtained in the last step. You should select the parameters value for noise type and window size.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)
row0_spacer1, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .05, 2.0, 0.1))

with row0_2:
    noise_method = st.selectbox(
        "Noise method", ("black", "white", "gaussian")
    )

with row0_3:
    window_size = st.slider(label="Window size", max_value=100, min_value=30, step=10)

if bg_image:
    target = np.array(logits_raw).argmax()
    output_predictions, heatmap, gen_batch = analysis(
        st.session_state["model"], 
        img, 
        target, 
        att, 
        order='crescent', 
        window_size=window_size, 
        noise_method=noise_method, 
        cumulative=True
        )
    output_us = get_u_positions(output_predictions, target)

    print('\tOutput predictions')
    print('\t', output_predictions.round(2))
    print('OUTPUT OF U ANALYSIS')
    print(output_us)

    row0_spacer0, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .05, 2.0, 0.1))

    with row0_2:
        heatmap[0] = (heatmap[0] - heatmap[0].min())/(heatmap[0].max() - heatmap[0].min())
        fig = go.Figure()

        fig.add_trace(go.Image(z=img))
        fig.add_trace(go.Heatmap(z=heatmap[0], zmin=0, zmax=1, colorscale="Jet", opacity=0.5))
        fig.update_layout(
            title="Importance of each input grid for model prediction", 
            coloraxis_showscale=False, 
            margin= dict(l=0,r=0,b=0),
            font=dict(size=18, color="Black")
            )

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with row0_3:
        fig = make_subplots(
            rows=2, 
            subplot_titles=(
                'Category predicted by each U step',
                'Target probability by each U step'
            )
            )
        
        fig.add_scatter(
            y=output_predictions.argmax(1),
            row=1,
            col=1,
            name="Category"
        )
        fig.add_scatter(
            y=output_predictions[:, target],
            row=2,
            col=1,
            name="Probability"
        )

        if len(output_us) > 0:
            y_argmax = output_predictions.argmax(1)
            idx_left, middle, idx_right = output_us[0]
            fig.add_scatter(
                y=[y_argmax[idx_left], y_argmax[middle], y_argmax[idx_right]],
                x=[idx_left, middle, idx_right], text=['Left', 'Middle', 'Right'], row=1, col=1, name="",
                marker=dict(size=[10, 10, 10], color=['green', 'red', 'green']),
                mode="markers+text", textposition="top center", showlegend=False)


        fig.update_layout(margin= dict(l=0,r=0,b=0, t=50.0))
        fig['layout']['xaxis']['title'] = 'U Step'
        fig['layout']['xaxis2']['title'] = 'U Step'
        fig['layout']['yaxis']['title'] = 'Category'
        fig['layout']['yaxis2']['title'] = 'Probability (%)'

        fig.update_layout(
            font=dict(size=10, color="Black")
        )
        fig.update_annotations(font_size=20)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    if len(output_us) > 0:
        st.write('Counter-intuitive behaviour: All the content from the right image is present on the middle image, but they have different class prediction.')
        idx_left, middle, idx_right = output_us[0]
        class_left = output_predictions.argmax(1)[idx_left]
        class_middle = output_predictions.argmax(1)[middle]
        class_right = output_predictions.argmax(1)[idx_right]

        print(gen_batch[idx_left].shape)

        fig = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=(f"U-Step {idx_left} - Class predicted {class_left}", f"U-Step {middle} - Class predicted {class_middle}", f"U-Step {idx_right} - Class predicted {class_right}")
            )

        fig.add_trace(
            go.Image(
                z=(gen_batch[idx_left].permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8),
                name='Left image'
            ),
            row=1, 
            col=1
            )
        
        fig.add_trace(
            go.Image(
                z=(gen_batch[middle].permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8),
                name='Middle image'
            ),
            row=1, 
            col=2
            )
        
        fig.add_trace(
            go.Image(
                z=(gen_batch[idx_right].permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8),
                name='Right image',
            ),
            row=1, 
            col=3
            )

        fig.update_layout(
            font=dict(
                size=30,
                color="Black"
            ),
            xaxis=dict(title='X Axis Title')
        )
        fig.update_xaxes(
            title='Left input',
            title_font_size=25,
            title_font_color='green',
            row=1,
            col=1
        )
        fig.update_xaxes(
            title='Middle input',
            title_font_size=25,
            title_font_color='red',
            row=1,
            col=2
        )
        fig.update_xaxes(
            title='Right input',
            title_font_size=25,
            title_font_color='green',
            row=1,
            col=3
        )
        fig.update_annotations(font_size=20)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

