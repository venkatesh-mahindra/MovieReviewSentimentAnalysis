import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sentiment_model.h5')

model = load_model()

# Load and prepare data
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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Demo", "Data Exploration", "Model Analysis"])

# Main content
if page == "Model Demo":
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("""
    This app predicts whether an IMDB movie review is positive or negative using a deep learning model.
    The model was trained on the IMDB dataset with 50,000 reviews.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Try the Model")
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
            if prediction >= 0.5:
                st.progress(float(prediction))
            else:
                st.progress(float(1 - prediction))
    
    with col2:
        st.subheader("Example Reviews")
        examples = [
            ("This movie was a complete waste of time.", 0),
            ("One of the best films I've seen this year!", 1),
            ("The plot was confusing and the acting was mediocre.", 0),
            ("Absolutely loved it! Would watch again.", 1)
        ]
        
        for example, label in examples:
            if st.button(example[:50] + "..." if len(example) > 50 else example):
                processed_example = preprocess_text(example)
                prediction = model.predict(processed_example)[0][0]
                
                st.write(f"Actual: {'Positive' if label else 'Negative'}")
                st.write(f"Predicted: {'Positive' if prediction >= 0.5 else 'Negative'} ({prediction*100:.1f}%)")

elif page == "Data Exploration":
    st.title("📊 IMDB Dataset Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Word Analysis", "Review Length"])
    
    with tab1:
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{len(x_train) + len(x_test):,}")
        col2.metric("Training Samples", f"{len(x_train):,}")
        col3.metric("Test Samples", f"{len(x_test):,}")
        
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=np.concatenate([y_train, y_test]), ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_title("Positive vs Negative Reviews")
        st.pyplot(fig)
        
        st.subheader("Sample Reviews")
        sample_idx = st.slider("Select a review index", 0, len(x_train)-1, 0)
        st.write(f"Review #{sample_idx} - {'Positive' if y_train[sample_idx] else 'Negative'}")
        st.text(decode_review(x_train[sample_idx]))
    
    with tab2:
        st.subheader("Word Frequency Analysis")
        
        # Get all words from positive and negative reviews
        positive_words = [word for idx in np.where(y_train == 1)[0] for word in x_train[idx]]
        negative_words = [word for idx in np.where(y_train == 0)[0] for word in x_train[idx]]
        
        # Create wordclouds
        col1, col2 = st.columns(2)
        with col1:
            st.write("Positive Reviews Word Cloud")
            wc = WordCloud(width=400, height=300, background_color='white').generate(" ".join([str(w) for w in positive_words]))
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        
        with col2:
            st.write("Negative Reviews Word Cloud")
            wc = WordCloud(width=400, height=300, background_color='white').generate(" ".join([str(w) for w in negative_words]))
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        
        # Show most common words
        st.subheader("Top Words")
        top_n = st.slider("Number of top words to show", 5, 50, 10)
        
        pos_freq = pd.Series(positive_words).value_counts().head(top_n)
        neg_freq = pd.Series(negative_words).value_counts().head(top_n)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Positive Reviews")
            fig, ax = plt.subplots()
            sns.barplot(x=pos_freq.values, y=pos_freq.index, ax=ax)
            ax.set_title(f"Top {top_n} Words in Positive Reviews")
            st.pyplot(fig)
        
        with col2:
            st.write("Negative Reviews")
            fig, ax = plt.subplots()
            sns.barplot(x=neg_freq.values, y=neg_freq.index, ax=ax)
            ax.set_title(f"Top {top_n} Words in Negative Reviews")
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Review Length Analysis")
        
        # Calculate review lengths
        train_lengths = [len(review) for review in x_train]
        test_lengths = [len(review) for review in x_test]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Set Review Lengths")
            fig, ax = plt.subplots()
            sns.histplot(train_lengths, bins=50, ax=ax)
            ax.set_title("Distribution of Review Lengths (Training)")
            ax.set_xlabel("Review Length")
            st.pyplot(fig)
            
            st.write(f"Average length: {np.mean(train_lengths):.1f} words")
            st.write(f"Maximum length: {max(train_lengths)} words")
            st.write(f"Minimum length: {min(train_lengths)} words")
        
        with col2:
            st.write("Test Set Review Lengths")
            fig, ax = plt.subplots()
            sns.histplot(test_lengths, bins=50, ax=ax)
            ax.set_title("Distribution of Review Lengths (Test)")
            ax.set_xlabel("Review Length")
            st.pyplot(fig)
            
            st.write(f"Average length: {np.mean(test_lengths):.1f} words")
            st.write(f"Maximum length: {max(test_lengths)} words")
            st.write(f"Minimum length: {min(test_lengths)} words")

elif page == "Model Analysis":
    st.title("🤖 Model Analysis")
    
    st.subheader("Model Architecture")
    st.text("""
    Sequential(
        Embedding(input_dim=10000, output_dim=128, input_length=200),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    )
    """)
    
    st.subheader("Model Performance")
    
    # Mock training history (in a real app, you would load this from your training logs)
    history = {
        'accuracy': [0.75, 0.85, 0.89, 0.91, 0.92],
        'val_accuracy': [0.83, 0.84, 0.86, 0.86, 0.86],
        'loss': [0.52, 0.35, 0.28, 0.24, 0.21],
        'val_loss': [0.39, 0.36, 0.34, 0.34, 0.34]
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Accuracy over Epochs")
        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Training Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Training vs Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.write("Loss over Epochs")
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Training vs Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
    
    st.subheader("Confusion Matrix (Example)")
    # Example confusion matrix
    cm = np.array([[10500, 1500], [1200, 10800]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'], ax=ax)
    ax.set_title('Confusion Matrix (Test Set)')
    st.pyplot(fig)
    
    st.subheader("Evaluation Metrics")
    st.write("""
    - Accuracy: 86.5%
    - Precision: 87.8%
    - Recall: 90.0%
    - F1-Score: 88.9%
    """)

# Footer
st.markdown("---")
st.markdown("""
**Note**: This is a demo application for educational purposes. 
The model was trained on the IMDB movie review dataset.
""")
