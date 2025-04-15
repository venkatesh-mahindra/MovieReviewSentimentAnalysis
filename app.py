import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 200

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sentiment_model.h5")
    return model

# Load word index
@st.cache_data
def get_word_index():
    word_index = imdb.get_word_index()
    index_word = {index + 3: word for word, index in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index

word_index = get_word_index()
model = load_model()

# Preprocess input
def preprocess(text):
    words = text.lower().split()
    encoded = [1]  # <START> token
    for word in words:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)  # <UNK>
    padded = pad_sequences([encoded], maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# Predict sentiment
def predict_sentiment(text):
    processed = preprocess(text)
    pred = model.predict(processed)[0][0]
    return pred

# Streamlit UI
st.set_page_config(page_title="🎬 IMDB Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below to check whether it's **Positive** or **Negative**.")

review_input = st.text_area("Write your review here", height=200)

if st.button("Analyze"):
    if review_input.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        with st.spinner("Analyzing..."):
            score = predict_sentiment(review_input)
            if score >= 0.5:
                st.success(f"👍 Positive Sentiment ({score:.2f})")
                st.image("https://i.imgur.com/8bI0nFQ.png", width=200)  # Happy face
            else:
                st.error(f"👎 Negative Sentiment ({score:.2f})")
                st.image("https://i.imgur.com/Ql5zN7U.png", width=200)  # Sad face

        st.markdown("### Sentiment Score Breakdown")
        st.progress(score)

st.markdown("---")
st.caption("Built with TensorFlow and Streamlit by [Venkatesh Mahindra✨]")
