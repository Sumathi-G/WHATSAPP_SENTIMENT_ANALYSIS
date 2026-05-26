import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="WhatsApp Chat Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------

st.title("💬 WhatsApp Chat Sentiment Analyzer")
st.write("### Team : 4")
st.write("RNN Basics / LSTMs & GRUs")

st.write("""
This app predicts whether a WhatsApp/group chat message is:

- 😊 Positive
- 😐 Neutral
- 😞 Negative
""")

# -------------------------------
# Load Model
# -------------------------------

model = load_model("sentiment_analysis_model.h5")

# -------------------------------
# Load Tokenizer
# -------------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Maximum sequence length
max_len = 50

# -------------------------------
# User Input
# -------------------------------

user_input = st.text_area(
    "Enter WhatsApp Message",
    placeholder="Type message here..."
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a message")

    else:

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_input])

        # Padding
        padded = pad_sequences(
            sequence,
            maxlen=max_len,
            padding='post'
        )

        # Prediction
        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        sentiments = {
            0: "Negative 😞",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        # Show result
        st.success(
            f"Predicted Sentiment: {sentiments[predicted_class]}"
        )

        # Show probabilities
        st.write("### Prediction Probabilities")
        st.write(prediction)

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.write("Built using Streamlit + TensorFlow + LSTM")