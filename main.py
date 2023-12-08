import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'Male', '1': 'Female'}

st.title('Name Prediction')

input = open('lrc_xray.pkl', 'rb')
model = pkl.load(input)

st.header('Upload image')
image = st.file_uploader('Choose an image', type=(['png', 'jpg', 'jpeg']))
                         
if image is not None:
  image = Image.open(image)
  st.image(image, caption='Test image')

  if st.button('Predict'):
    image = image.resize((227*227*3, 1))
    vector = np.array(image)
    label = str((model.predict(vector))[0])

    st.header('Result')
    st.text(class_list[label])
