# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model, Sequential
# import streamlit as st

# #load imdb dataset and word index
# word_index = imdb.get_word_index()
# word_index 
# # joining the index and words to see sentence meaning f
# reverse_word_index = {value: key for (key, value) in word_index.items()}

# # load the trained model
# try:
#     model = load_model('RNN/simple_rnn_imdb_model.h5')
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# # helper functions
# # to decode the review
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
# # function to prepreocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
#     padded = sequence.pad_sequences([encoded], maxlen=500)
#     return padded

# # prediction function

# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     prediction = model.predict(processed_text)
#     sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
#     return sentiment, prediction[0][0]

# # Streamlit app
# st.title("IMDB Movie Review Sentiment Analysis")
# st.write("Enter a movie review to predict its sentiment (positive/negative).")

# # User input
# user_input = st.text_area("Movie Review:")

# if st.button("Classify"):
#     sentiment, confidence = predict_sentiment(user_input)
#     st.write(f"Predicted Sentiment: **{sentiment}**")
#     st.write(f"Confidence: **{confidence:.2f}**")
# else:
#     st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os

# -------------------------
# Caching heavy resources
# -------------------------

@st.cache_data(show_spinner=True)
def get_word_index():
    return imdb.get_word_index()

@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    model_path = os.path.join("RNN", "simple_rnn_imdb_model.h5")
    return load_model(model_path)

# -------------------------
# Helper functions
# -------------------------

def decode_review(encoded_review, reverse_index):
    """Convert encoded review to words"""
    return ' '.join([reverse_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text, word_index):
    """Convert user text to padded sequence for model"""
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

def predict_sentiment(text, model, word_index):
    processed = preprocess_text(text, word_index)
    prediction = model.predict(processed, verbose=0)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, float(prediction[0][0])

# -------------------------
# Streamlit App
# -------------------------

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative).")

# Load model and word index (cached)
word_index = get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}
model = load_sentiment_model()

# User input
user_input = st.text_area("Movie Review:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        sentiment, confidence = predict_sentiment(user_input, model, word_index)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence: **{confidence:.2f}**")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")
