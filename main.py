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
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model, Sequential
import streamlit as st

# Load IMDB dataset and word index
@st.cache_data(show_spinner=True)
def load_word_index():
    return imdb.get_word_index()

word_index = load_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the trained model
@st.cache_data(show_spinner=True)
def load_rnn_model():
    try:
        model = load_model('RNN/simple_rnn_imdb_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_rnn_model()
if model is not None:
    st.success("Model loaded successfully!")

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

MAX_WORDS = 10000  # vocab size used during model training

def preprocess_text(text):
    words = text.lower().split()
    encoded = []
    for w in words:
        idx = word_index.get(w, 2) + 3  # 2 = unknown
        if idx >= MAX_WORDS:
            idx = 2
        encoded.append(idx)
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

# Prediction function
@st.cache_data(show_spinner=True)
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text, verbose=0)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, float(prediction[0][0])

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative).")

user_input = st.text_area("Movie Review:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence: **{confidence:.2f}**")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")
