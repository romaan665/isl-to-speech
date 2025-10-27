import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
from playsound import playsound
import os
import uuid
import threading
import time

# -------------------------------
# Load Model
# -------------------------------
model = load_model("action.h5")

# -------------------------------
# Actions and Translations
# -------------------------------
actions = ['help', 'blood', 'pain', 'bone_break', 'fever']

spoken_map = {
    'help': 'कृपया मेरी मदद करें!',
    'blood': 'मुझे खून निकल रहा है!',
    'pain': 'मुझे दर्द हो रहा है!',
    'bone_break': 'मेरी हड्डी टूट गई है!',
    'fever': 'मुझे बुखार है!'
}

# -------------------------------
# Mediapipe Setup
# -------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# Global State
# -------------------------------
_speaking = False

# -------------------------------
# Session state for history
# -------------------------------
if 'recognized_history' not in st.session_state:
    st.session_state.recognized_history = []

# -------------------------------
# Extract Keypoints
# -------------------------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# -------------------------------
# Hindi Text-to-Speech (non-blocking)
# -------------------------------
def speak_text_hi(text):
    global _speaking
    if _speaking:
        return
    def run():
        global _speaking
        _speaking = True
        fname = f"tts_{uuid.uuid4().hex}.mp3"
        try:
            gTTS(text=text, lang='hi').save(fname)
            playsound(fname)
        except Exception as e:
            print("TTS error:", e)
        finally:
            if os.path.exists(fname):
                os.remove(fname)
            _speaking = False
    threading.Thread(target=run, daemon=True).start()

# -------------------------------
# Mediapipe Detection Function
# -------------------------------
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

# -------------------------------
# Draw Colored Landmarks
# -------------------------------
def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🖐️ ISL Emergency Sign to Hindi Speech")
start_camera = st.button("Start Real-Time Detection")

history_placeholder = st.empty()
sequence, sentence, predictions = [], [], []
threshold = 0.7
STABILITY_FRAMES = 10
COOLDOWN_SECS = 3
last_spoken_time = {}
last_word = None

if start_camera:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Check if any hands are visible
            hands_visible = results.left_hand_landmarks or results.right_hand_landmarks

            # Collect sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Predict only if 30 frames and hands visible
            if len(sequence) == 30 and hands_visible:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(res))
                conf = float(res[pred_idx])
                predictions.append(pred_idx)

                if len(predictions) > STABILITY_FRAMES:
                    predictions = predictions[-STABILITY_FRAMES:]

                # Majority vote + stability
                if conf > threshold:
                    most_common = max(set(predictions), key=predictions.count)
                    freq = predictions.count(most_common)

                    word = actions[most_common]
                    now = time.time()

                    if word != last_word:
                        if word not in last_spoken_time or (now - last_spoken_time[word]) > COOLDOWN_SECS:
                            sentence.append(word)
                            speak_text_hi(spoken_map.get(word, word))
                            last_spoken_time[word] = now
                            last_word = word
                            st.session_state.recognized_history.append(word)
            else:
                # No sign detected
                cv2.putText(image, "No sign detected", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                predictions = []
                last_word = None

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Draw sentence bar
            cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
            cv2.putText(image, ' '.join(sentence), (8,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Show live image
            stframe.image(image, channels="BGR", use_column_width=True)

            # Show recognized history
            history_placeholder.markdown("### Recognized Signs History")
            history_placeholder.write(st.session_state.recognized_history)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
