
import numpy as np, pandas as pd, streamlit as st
import os
from PIL import Image
from pprint import pprint
from termcolor import colored, cprint

from functions import *

st.title(body='IntelliVision')
st.markdown(body='#Crystal clear awareness in robotic vision.')

# Classify Image
st.markdown(body='##Classify Image')

img_upload = st.file_uploader(
    label='Upload Image',
    type=np.asarray(a=list(['.jpg', '.jpeg', '.png']))
)

st.write(img_upload)
st.write(type(img_upload))


img_path = './images/english_village.png'
img_pil = Image.open(fp=img_path, mode='r')
img = st.image(
    image=img_path,
    caption='caption of image',
    use_column_width=True
)

clf_model = st.selectbox(
    label='Select Model',
    options=np.asarray(a=list([
        'MobileNetV2',
        'ResNet50',
        'InceptionV3',
        'DenseNet121'
    ]))
)

# classify_image(img_path=)

st.button(label='Classify', onclick=st.write(text='Button clicked'))
