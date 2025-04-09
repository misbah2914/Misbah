import cv2
from deepface import DeepFace

def capture_face():
    cap = cv2.VideoCapture(0)  # Open webcam
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("face.jpg", frame)  # Save the image
        cap.release()
        return "face.jpg"

    else:
        cap.release()
        return None

def analyze_expression(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['emotion'])
        emotion = analysis[0]['dominant_emotion']
        return f"Detected Emotion: {emotion}"
    except Exception as e:
        return f"Error: {str(e)}"

# Run the facial analysis
image_path = capture_face()
if image_path:
    result = analyze_expression(image_path)
    print(result)
else:
    print("Failed to capture face.")