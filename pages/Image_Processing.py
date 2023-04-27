import streamlit as st
from PIL import Image
import numpy as np
import copy

from image_processing import texture, structure
from model_utils import inference

import plotly.express as px

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
        "type":['Original'] * len(logits_raw) + ['Transformed'] * len(logits_raw)
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
    # TO DO: Validate mean and std format
    st.session_state['set_mean'] = eval(set_mean)
    st.session_state['set_std'] = eval(set_std)
else:
    bg_image = None


st.header('Texture Analysis')
st.markdown("""This group implements image transformations that change the input image texture by adding noise. The results obtained from these functions allow the user to get insights into the model's dependence on texture patterns and the model's robustness to noise.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)

options = ["GaussianNoise", "ShotNoise", "ImpulseNoise", "SpekleNoise"]

row0_spacer2, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .1, 2.0, .1))

with row0_2:
    texture_option = st.selectbox(label="Select texture transforming", options=options, index=0)

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
    row0_spacer1, row0_2, row0_spacer3, row_03, row0_spacer4 = st.columns((.1, 2, .1, 2, .1))

    img = Image.open(bg_image).resize((set_img_size, set_img_size))
    vec = np.asarray(img)
    height, width, channels = vec.shape

    with row0_2:
        ans = texture_class(**args)(vec)
        texture_pilimg = Image.fromarray(ans.astype(np.uint8), mode='RGB')
        st.image(texture_pilimg, use_column_width=True)
        print(ans.shape, ans.max(), ans.min())

    with row_03:
        logits_raw = inference.Predictor()(st.session_state["model"], img)
        logits_txt = inference.Predictor()(st.session_state["model"], texture_pilimg)

        df = build_dataframe_pair(logits_raw, logits_txt)

        fig = px.bar(
            df,
            x='categories',
            y='probabilities',
            color="type",
            barmode="group",
            title="Original vs Transformer image output"
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


st.header('Structure Analysis')
st.markdown("""This analysis implements three different spatial transformations (e.g., Patch Shuffle, Horizontal Shuffle, Vertical Shuffle) that deform the structure of the objects in the scene but keep the original texture and color information. These transformations can be useful to infer insights about the model dependency on object's shape or texture bias.""")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)

options = ["PatchShuffle", "HorizontalShuffle", "VerticalShuffle"]

row0_spacer2, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((0.1, 2.0, 0.1, 2.0, 0.1))

with row0_2:
    structure_option = st.selectbox(label="Select texture transforming", options=options, index=0)

with row0_3:

    structure_class = getattr(structure, structure_option)
    parameters = getattr(structure_class, "parameters")
    args = {}

    if parameters.get('patch_size', False):
        patchsize = st.select_slider('Patch Size', [2, 4, 8, 16])
        args['patch_size'] = patchsize

    if parameters.get('lam_weight', False):
        lam_weight = st.slider('Lam weight', 0, 100, 1)
        args['lam_weight'] = lam_weight

    if parameters.get('amount', False):
        amount = st.slider('Amount', 0.0, 0.30, 0.03)
        args['amount'] = amount

if bg_image:
    row0_spacer1, row0_2, row0_spacer3, row_03, row0_spacer4 = st.columns((.1, 2, .1, 2, .1))

    img = Image.open(bg_image).resize((set_img_size, set_img_size))
    vec = np.asarray(img)
    height, width, channels = vec.shape

    with row0_2:
        ans = structure_class(**args)(img)
        texture_pilimg = Image.fromarray(ans.astype(np.uint8), mode='RGB')
        st.image(texture_pilimg, use_column_width=True)

    with row_03:
        logits_raw = inference.Predictor()(st.session_state["model"], img)
        logits_txt = inference.Predictor()(st.session_state["model"], texture_pilimg)

        df = build_dataframe_pair(logits_raw, logits_txt)

        fig = px.bar(
            df,
            x='categories',
            y='probabilities',
            color="type",
            barmode="group",
            title="Original vs Structure analysis output"
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