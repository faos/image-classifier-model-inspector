import streamlit as st  # data web app development
from model_utils import loader, utils

if "shared" not in st.session_state:
   st.session_state["shared"] = True

st.set_page_config(
    layout="wide", 
    #page_icon='images/logo_di2win3.png',
    page_title='Local Analysis'
    )


st.header('Introduction')
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)
st.markdown("Model inspector is a tool that enables users to perform a robustness analysis of image classification models. The user may load an image classifier model and an input image, then perform several image transformations to evaluate the model behavior around the input image. Besides, the user may apply three different category transformations in the input image to assess the model behavior: image processing, interpretability-based, and signal.")
st.markdown(
"""
**Modules**:
- Image processing
- Interpretability
- Signal
"""
)

st.header("Load Model")
st.markdown("""<hr style="width:100%; text-align:left; margin:0">""", unsafe_allow_html=True)

framework_options = ["", "pytorch", "timm"]

framework_option = st.selectbox(label="Framework", options=framework_options, index=0)
options = utils.get_framework_options(framework_option)

model_option = st.selectbox(label="Select model architecture", options=options, index=0)
num_categories = st.number_input(label="Number of categories", step=1, value=50)

weights_file = st.file_uploader("Weights:", type=[".pth"])


print('framework_option:\t', framework_option)
print("Model option:\t", model_option)
print("Num categories:\t", num_categories)

if weights_file:
    print('Loading weights:', weights_file)
    ml_loader = loader.ModelLoader(
        framework_option, 
        model_name=model_option, 
        num_categories=num_categories,
        weight_path=weights_file
        )
else:
    ml_loader = loader.ModelLoader(
        framework_option, 
        model_name=model_option, 
        num_categories=num_categories,
        )

st.session_state["model"] = ml_loader.model

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"], key="image-input")

if st.session_state.get('img_size', False):
    set_img_size = st.sidebar.select_slider(label="Select image size", value=st.session_state['img_size'], options=[32, 96, 224, 299])
else:
    set_img_size = st.sidebar.select_slider(label="Select image size", value=224, options=[32, 96, 224, 299])

if st.session_state.get("set_mean", False):
    set_mean = st.sidebar.text_input(label="Mean", value=st.session_state['set_mean'])
else:
    set_mean = st.sidebar.text_input(label="Mean", value="(0.485, 0.456, 0.406)")

if st.session_state.get("set_std", False):
    set_std = st.sidebar.text_input(label="Std", value=st.session_state['set_std'])
else:
    set_std = st.sidebar.text_input(label="Std", value="(0.229, 0.224, 0.225)")

st.session_state['img_size'] = set_img_size
st.session_state['set_mean'] = eval(set_mean)
st.session_state['set_std'] = eval(set_std)
st.session_state['num_classes'] = num_categories

if bg_image:
    st.session_state["input-image-file"] = bg_image


