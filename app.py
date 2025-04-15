import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sentiment_model.h5')

model = load_model()

# Load data
@st.cache_data
def load_data():
    vocab_size = 10000
    max_len = 200
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    return (x_train, y_train), (x_test, y_test), word_index, vocab_size, max_len

(x_train, y_train), (x_test, y_test), word_index, vocab_size, max_len = load_data()

# Decode review function
def decode_review(encoded_review):
    index_word = {index + 3: word for word, index in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"
    return ' '.join([index_word.get(i, '?') for i in encoded_review])

# Preprocess text input
def preprocess_text(text):
    word_to_index = {word: (index + 3) for word, index in word_index.items()}
    word_to_index["<PAD>"] = 0
    word_to_index["<START>"] = 1
    word_to_index["<UNK>"] = 2
    
    tokens = text.lower().split()
    encoded = [1]  # Start token
    for token in tokens:
        encoded.append(word_to_index.get(token, 2))  # 2 is UNK token
    padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')
    return padded

# App interface
st.title("🎬 IMDB Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review:", 
                         "This movie was fantastic! The acting was superb and the plot kept me engaged throughout.")

if st.button("Predict Sentiment"):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)[0][0]
    
    st.subheader("Prediction Result")
    if prediction >= 0.5:
        st.success(f"Positive sentiment ({prediction*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"Negative sentiment ({(1-prediction)*100:.1f}%)")
    
    # Show confidence meter
    st.write("Confidence:")
    st.progress(float(prediction if prediction >= 0.5 else 1 - prediction))

# Basic data visualization
if st.checkbox("Show dataset info"):
    st.subheader("Dataset Information")
    st.write(f"Training samples: {len(x_train)}")
    st.write(f"Test samples: {len(x_test)}")
    
    # Simple histogram of review lengths
    review_lengths = [len(review) for review in x_train[:1000]]  # First 1000 for performance
    fig, ax = plt.subplots()
    ax.hist(review_lengths, bins=50)
    ax.set_title("Review Length Distribution")
    ax.set_xlabel("Length (words)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
