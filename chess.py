import streamlit as st
import numpy as np
import tensorflow as tf
import PIL


st.title("Chess Piece Classifier")
st.write("### This model takes as input the image of a chess piece and tells you what piece it is")
model = tf.keras.models.load_model('model/content/sample_data/model')
file = st.file_uploader('Upload an image file')

classes = ["bishop","king","pawn","knight","rook","queen"]

if file is not None:
    img = PIL.Image.open(file)
    img = img.resize((85,85))
    img = img.convert('RGB')
    img = np.asarray(img)
    out = tf.nn.softmax(model.predict(np.array([img])))
    prediction = np.argmax(out)
    st.write(f'### The piece was found to be a {classes[prediction]} ')
