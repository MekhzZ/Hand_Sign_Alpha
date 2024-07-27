import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained SVM model, scaler, and label encoder
model = joblib.load('svm_hand_sign_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process frames and make predictions
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Flatten the list
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Scale the landmarks
            landmarks = scaler.transform(landmarks)

            # Predict the hand sign
            prediction = model.predict(landmarks)
            hand_sign = label_encoder.inverse_transform(prediction)[0]  # Decode the prediction

            # Display the prediction on the frame
            cv2.putText(frame, hand_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Streamlit app
st.title('Real-Time Hand Sign Detection')
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    FRAME_WINDOW.image(frame)

cap.release()
