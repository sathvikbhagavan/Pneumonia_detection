import tensorflow as tf
import streamlit as st
import numpy as np
from io import BytesIO
import io
import cv2


model = tf.keras.models.load_model('pneumonia_detection.h5')
print('Loaded model from disk')

def preprocess(image):
    x = tf.convert_to_tensor(image)
    x = tf.image.resize(x, (200,200))
    x = x / 255.0
    return x

def main():
    st.title('Pneumonia Detection')
    st.subheader('This is a web application to test from x-ray image whether the patient has pneumonia or not')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file = st.file_uploader('Please upload an image file', type=['jpeg'])
    show_file = st.empty()
    if not file:
        show_file.info('Please upload a file in jpg format')
        return
    content = file.getvalue()
    image = cv2.imdecode(np.fromstring(content, dtype=np.uint8), 1)
    image = cv2.resize(image, (600, 400))
    show_file.image(image, 'Input Image')
    image = image.reshape(1, image.shape[0], image.shape[1], 3)
    image = preprocess(image)
    output = model(image)
    if output[0,0] <= 0.5:
        st.subheader('The patient is Normal!')
        st.write('The probability of Pneumonia is {:0.2f}'.format(output[0,0]))
    else:
        st.subheader('The patient has Pneumonia')
        st.write('The probability of Pneumonia is {:0.2f}'.format(output[0,0]))

main()

