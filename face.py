import streamlit as st
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
import os

# Load SVM Model
def load_svm_model(model_path):
    return joblib.load(model_path)

# Get face embedding
def get_embedding(model, face):
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    return model.embeddings(face)[0]

# Recognize face in live feed
def recognize_face_live(facenet_model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    # Load stored face embeddings
    save_path = "face_embeddings.pkl"
    if os.path.exists(save_path):
        data = joblib.load(save_path)
    else:
        data = {}
    
    faces = []  # Initialize faces to avoid UnboundLocalError
    
    if st.button("Recognize Face", key="recognize_face_btn"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            recognized_name = "Unknown"
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                embedding = get_embedding(facenet_model, face)
                
                # Compare embedding with stored data
                min_dist = float('inf')
                for name, stored_embedding in data.items():
                    dist = np.linalg.norm(stored_embedding - embedding)
                    if dist < min_dist:
                        min_dist = dist
                        recognized_name = name if dist < 1.0 else "Unknown"
                
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            stframe.image(frame, channels="BGR")
        cap.release()
    
    cap.release()
    return recognized_name if len(faces) > 0 else "No face detected"

# Capture face and store embedding
def capture_face(name, facenet_model):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    if st.button("Capture Face", key="capture_face_btn"):
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
        else:
            stframe.image(frame, channels="BGR")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                embedding = get_embedding(facenet_model, face)
                
                save_path = "face_embeddings.pkl"
                if os.path.exists(save_path):
                    data = joblib.load(save_path)
                else:
                    data = {}
                
                data[name] = embedding
                joblib.dump(data, save_path)
                st.success(f"Face of {name} captured and saved!")
            else:
                st.error("No face detected. Try again.")
        
        cap.release()

# Main function
def main():
    st.set_page_config(page_title="Face Recognition App", layout="wide")
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Modern Face Recognition App")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Select Option", ["Recognize Face", "Capture & Register Face"])
    
    facenet_model = FaceNet()
    
    if choice == "Recognize Face":
        st.write("Position your face in front of the camera and press 'Recognize Face'.")
        recognized_name = recognize_face_live(facenet_model)
        st.write(f"Recognized: {recognized_name}")
    
    elif choice == "Capture & Register Face":
        name = st.text_input("Enter Name:")
        if name:
            st.write("Press the button below to capture a new face.")
            capture_face(name, facenet_model)

if __name__ == "__main__":
    main()
