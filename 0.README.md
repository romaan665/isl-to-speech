# isl-to-speech
# 🩺 ISL to Speech Conversion System for Medical Emergencies

A **real-time Indian Sign Language (ISL) recognition system** designed to assist **speech- and hearing-impaired individuals during medical emergencies**.  
The system detects ISL gestures through a webcam and converts them into **spoken Hindi or Marathi** using deep learning and text-to-speech technology.  
Built with **MediaPipe**, **TensorFlow (LSTM)**, and **Streamlit**, this project bridges critical communication gaps in emergency healthcare settings 🏥💫

---

## ⚙️ Features

- 🖐️ Real-time ISL gesture recognition using webcam input  
- 🧩 LSTM model trained on medical/emergency-related gestures  
- 🗣️ Text-to-Speech (gTTS) conversion to speak recognized signs in Hindi/Marathi  
- 💻 Streamlit-based web app for quick and interactive response  
- ⏱️ Aimed at enabling **faster communication in emergency situations**  

---

## 🧩 Libraries and Dependencies

| Library / Package | Purpose / Description |
|--------------------|-----------------------|
| tensorflow | Deep learning framework for LSTM model creation and training |
| mediapipe | Real-time pose, hand, and face landmark detection |
| opencv-python | Camera feed handling, frame processing, and visualization |
| numpy | Numerical operations and data manipulation |
| matplotlib | Visualization of training accuracy and sample outputs |
| scikit-learn | Dataset splitting, accuracy calculation, and metrics |
| gtts | Google Text-to-Speech conversion for Hindi audio |
| playsound | Playback of generated speech files |
| os, uuid, threading, time | File handling, parallel speech playback, and timing control |
| streamlit | Interactive interface for starting real-time gesture detection |

---

## 🗂️ Project Structure
├── app.py → Streamlit-based interface for real-time ISL detection and speech conversion
├── signlanguage.ipynb → Model training and testing notebook
├── requirements.txt → Required libraries and dependencies
├── Project_Documentation.docx → Detailed project documentation
├── images/
│ ├── flowchart.png → System flowchart visualization
│ └── sample_output.png → Example output from gesture detection


---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/romaan665/isl-to-speech.git
2. **Navigate to the project folder:**
   cd ISL-to-Speech
   
3. **Install dependencies:**
   pip install -r requirements.txt

4. **Run the Streamlit app:**
   streamlit run app.py

5. **The live interface will open in your browser — start showing signs to get instant speech output 🔊**

💡 **Future Enhancements**

Extend gesture vocabulary for broader emergency use cases

Add multi-language and voice customization options

Integrate with IoT devices or hospital alert systems
