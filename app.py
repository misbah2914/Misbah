import streamlit as st
import cv2
import mediapipe as mp
import speech_recognition as sr
from deepface import DeepFace
from textblob import TextBlob
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to analyze speech
def transcribe_speech():
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_audio is not None:
    recognizer = sr.Recognizer()
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_audio is not None:
    recognizer = sr.Recognizer()
    with sr.AudioFile(uploaded_audio) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
        except sr.UnknownValueError:
            st.write("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.write("Speech Recognition service is unavailable.")
        try:
            text = recognizer.recognize_google(audio)
            return text
        except:
            return "Speech not recognized."

def analyze_speech(text):
    filler_words = ["um", "uh", "like", "you know", "so"]
    words = text.lower().split()
    filler_count = sum(1 for word in words if word in filler_words)

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 (negative) to +1 (positive)

    feedback = f"Sentiment Score: {sentiment:.2f}. "
    feedback += f"Try to reduce filler words ({filler_count} detected)." if filler_count > 0 else "Good clarity!"
    return feedback

# Function to analyze emotions
def analyze_expression(image):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except:
        return "Emotion detection failed"

# Function for posture detection
def analyze_posture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        head_y = landmarks[mp_pose.PoseLandmark.NOSE].y

        return "Good posture!" if head_y < shoulder_y else "Try to sit upright."
    return "No posture detected."

# Streamlit UI
st.title("🧠 AI-Powered Interview Coach")
st.sidebar.header("Choose an Option")

option = st.sidebar.radio("Select a Feature", ["Facial & Posture Analysis", "Speech Analysis"])

if option == "Facial & Posture Analysis":
    st.write("📷 Turn on your camera for real-time posture & emotion analysis.")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam.")
            break

        emotion = analyze_expression(frame)
        posture_feedback = analyze_posture(frame)

        # Display output
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, posture_feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        if st.sidebar.button("Stop Webcam"):
            break

    cap.release()

elif option == "Speech Analysis":
    st.write("🎤 Click below to start speaking.")
    if st.button("Start Speaking"):
        speech_text = transcribe_speech()
        speech_feedback = analyze_speech(speech_text)
        st.write("🗣 *You Said:*", speech_text)
        st.write("📊 Speech Feedback:", speech_feedback)
