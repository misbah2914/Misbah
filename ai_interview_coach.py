import cv2
import mediapipe as mp
import speech_recognition as sr
from deepface import DeepFace
from textblob import TextBlob

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Speech recognition function
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Speech not recognized."
        except sr.RequestError:
            return "Error connecting to speech recognition service."

# Speech analysis function
def analyze_speech(text):
    filler_words = ["um", "uh", "like", "you know", "so"]
    words = text.lower().split()
    filler_count = sum(1 for word in words if word in filler_words)
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
    
    feedback = f"Sentiment Score: {sentiment:.2f}. "
    feedback += f"Try to reduce filler words ({filler_count} detected)." if filler_count > 0 else "Good clarity!"
    return feedback

# Facial expression analysis function
def analyze_expression(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except:
        return "Emotion detection failed"

# Open the webcam for posture & facial analysis
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    feedback = "Analyzing posture..."
    
    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        head_y = landmarks[mp_pose.PoseLandmark.NOSE].y

        if head_y < shoulder_y:
            feedback = "Good posture!"
        else:
            feedback = "Try to sit upright."

    # Perform facial expression analysis
    emotion = analyze_expression(frame)
    
    # Display feedback
    cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Interview Coach", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Run speech analysis after closing the webcam
print("Starting Speech Analysis...")
speech_text = transcribe_speech()
speech_feedback = analyze_speech(speech_text)

print("\nFinal Analysis:")
Print(f"Speech Feedback:{speech_feedback}")