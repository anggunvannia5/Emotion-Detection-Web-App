import pickle
import numpy as np
import streamlit as st

# Load saved model and vectorizer
model = pickle.load(open('emotion_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

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
