import streamlit as st
import pickle as pkl
import numpy as np

class_list = {'0': 'Male', '1': 'Female'}

st.title('Name Prediction')
input = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input)

input = open('ec_vinames.pkl', 'rb')
model = pkl.load(input)

st.header('Write Name')
txt = st.text_area("","")
                         
if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.text(class_list[label])
