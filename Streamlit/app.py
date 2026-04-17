import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

from utils.database import supabase
from utils.model_loader import load_model, transform
from utils.face_utils import detect_face, get_embedding
from utils.recognition import add_embedding, load_database, recognize, log_attendance

# -------------------------
# TEST DB CONNECTION
# -------------------------
try:
    supabase.table("staff_embeddings").select("*").limit(1).execute()
    st.success("✅ Database Connected")
except Exception as e:
    st.error(f"❌ DB Error: {e}")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = get_face_cascade()

st.title("📸 Staff Attendance System")

mode = st.sidebar.selectbox("Mode", ["Register", "Recognize", "View Attendance"])

# -------------------------
# REGISTER
# -------------------------
if mode == "Register":

    name = st.text_input("Enter Name")
    img_file = st.camera_input("Capture Face")

    if img_file and name:

        frame = cv2.imdecode(
            np.frombuffer(img_file.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )

        img, _ = detect_face(frame, face_cascade)

        if img is None:
            st.warning("⚠ No face detected")
        else:
            emb = get_embedding(model, img, transform)

            # 🔥 store multiple embeddings
            for _ in range(5):
                add_embedding(name, emb)

            st.success("✅ Registered successfully")

# -------------------------
# RECOGNITION
# -------------------------
elif mode == "Recognize":

    run = st.checkbox("Start Camera")

    if run:

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        recent = deque(maxlen=5)

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            img, _ = detect_face(frame, face_cascade)

            if img is None:
                status_placeholder.info("No face detected")
                frame_placeholder.image(frame, channels="BGR")
                continue

            db = load_database()
            emb = get_embedding(model, img, transform)

            name, score = recognize(emb, db)

            recent.append(name)
            final_name = max(set(recent), key=recent.count)

            if final_name == "Unidentified Person":
                status_placeholder.warning("❌ Unidentified Person")
                color = (0, 0, 255)
            else:
                status_placeholder.success(f"✅ {final_name}")
                log_attendance(final_name)
                color = (0, 255, 0)

            cv2.putText(frame, f"{final_name} ({score:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

            frame_placeholder.image(frame, channels="BGR")

# -------------------------
# VIEW ATTENDANCE
# -------------------------
elif mode == "View Attendance":

    try:
        response = supabase.table("attendance").select("*").execute()
        df = pd.DataFrame(response.data)

        st.dataframe(df)

    except:
        st.info("No attendance yet")