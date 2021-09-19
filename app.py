import numpy as np, pandas as pd, os, streamlit as st
from PIL import Image
from pprint import pprint
from termcolor import colored, cprint

from functions import *

st.title(body='IntelliVision')

# Classify Image
st.markdown(body='## Classify Image')

img_upload = st.file_uploader(
    label='Upload Image',
    type=list(['jpg', 'png'])
)

clf_model = st.selectbox(
    label='Select Model',
    options=np.asarray(a=[
        'MobileNetV2',
        'ResNet50',
        'InceptionV3',
        'DenseNet121'
    ])
)

n_predictions = st.slider(
    label='Number of predictions.',
    min_value=1,
    max_value=10
)

def classify_callback():
    global img_pil, img_np
    if img_upload is None:
        st.write('No image uploaded yet.')
        return

    img_pil = Image.open(fp=img_upload)
    img_np = np.asarray(a=img_pil)[:, :, :3]

    st.image(
        image=img_upload,
        use_column_width=True
    )

    global predictions, probabilities
    predictions, probabilities = classify_image(img=img_np, model=clf_model, n_predictions=n_predictions)
    st.write(pd.DataFrame(data=dict({
        'Prediction': predictions,
        'Score': probabilities
    })))

    nums = dict()
    list1 = predictions
    list2 = probabilities
    for i in range(len(list1)):
        nums[list1[i]] = [np.zeros((i,1)), list2[i]]  
        temp = np.zeros((len(list1)))
        temp[i] = list2[i]
        nums[list1[i]] = temp
        
    chart_data = pd.DataFrame(nums)
    st.bar_chart(chart_data)

    st.success(body='Success')
    st.balloons()

if st.button(label='Classify'):
    classify_callback()

st.markdown(body='## Object Detection')

detector_model = st.selectbox(
    label='Select Model',
    options=np.asarray(a=[
        'RetinaNet',
        'AsYOLOv3',
        'TinyYOLOv3'
    ])
)

def detect_callback():
    global img_pil, img_np
    if img_upload is None:
        st.write('No image uploaded yet.')
        return

    img_pil = Image.open(fp=img_upload)
    img_np = np.asarray(a=img_pil)[:, :, :3]

    detections = detect_objects_from_image(img=img_np, model=detector_model)
    detected_objects = list([obj['name'] for obj in detections[0]])
    probabilities = list([obj['percentage_probability'] for obj in detections[0]])

    st.image(
        image='./detections/object_detection.png',
        use_column_width=True
    )

    st.write(pd.DataFrame(data=dict({
        'Detected Objects': detected_objects,
        'Scores': probabilities
    })))

    st.success(body='Success')
    st.balloons()

if st.button(label='Detect'):
    detect_callback()
