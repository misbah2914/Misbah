import speech_recognition as sr
from textblob import TextBlob

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            return None
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
            return None

def analyze_speech(text):
    if text is None:
        return "No valid speech input detected."

    filler_words = ["um", "uh", "like", "you know", "so"]
    words = text.lower().split()
    filler_count = sum(1 for word in words if word in filler_words)

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 (negative) to +1 (positive)

    feedback = f"Sentiment Score: {sentiment:.2f}. "
    feedback += f"Try to reduce filler words ({filler_count} detected)." if filler_count > 0 else "Good clarity!"

    return feedback

# Run the speech analysis
transcribed_text = transcribe_speech()
feedback = analyze_speech(transcribed_text)
print(feedback)