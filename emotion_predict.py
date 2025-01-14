import pickle
import numpy as np
import streamlit as st
import os

# Pastikan path file sesuai dengan struktur di GitHub
model_path = os.path.join(os.path.dirname(__file__), "emotion_model.sav")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.sav")

# Load model dan vectorizer
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define emotion labels
emotions = {
    0: "Neutral",
    1: "Happiness",
    2: "Excitement",
    3: "Calm",
    4: "Sadness",
    5: "Fatigue"
}

# Streamlit app setup
st.title("Emotion Detection Web App")

st.write("""
### Enter a sentence to detect its emotion
""")

# User input
user_input = st.text_input("Enter text", placeholder="Type your sentence here...")

# Prediction button
if st.button("Predict Emotion"):
    if user_input.strip():
        # Preprocess and predict
        input_data_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_data_vec)
        predicted_emotion = emotions.get(prediction[0], "Unknown")

        # Display result
        st.success(f"Predicted Emotion: {predicted_emotion}")

        # Additional details
        if prediction[0] == 4:
            st.info("The model predicts that the emotion is sadness.")
        elif prediction[0] == 1:
            st.info("The model predicts that the emotion is happiness.")
        else:
            st.info(f"The model predicts that the emotion is {predicted_emotion}.")
    else:
        st.error("Please enter a valid text input.")
