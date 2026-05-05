import streamlit as st
import numpy as np
import pickle
import sqlite3
import pandas as pd
from datetime import datetime
from PIL import Image
emotion = "neutral"
import os

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="AI Attendance System", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    svm_model     = pickle.load(open(os.path.join(BASE, "face_svm.pkl"), "rb"))
    label_encoder = pickle.load(open(os.path.join(BASE, "label_encoder.pkl"), "rb"))
    emotion_detector = FER()
    return svm_model, label_encoder, emotion_detector

def init_db():
    conn = sqlite3.connect(os.path.join(BASE, "attendance.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, emotion TEXT, date TEXT, time TEXT)""")
    conn.commit()
    conn.close()

def mark_attendance(name, emotion):
    conn   = sqlite3.connect(os.path.join(BASE, "attendance.db"))
    cursor = conn.cursor()
    now    = datetime.now()
    date   = now.strftime("%Y-%m-%d")
    time   = now.strftime("%H:%M:%S")
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO attendance VALUES (NULL,?,?,?,?)",
                       (name, emotion, date, time))
        conn.commit()
    conn.close()

def get_attendance():
    conn = sqlite3.connect(os.path.join(BASE, "attendance.db"))
    df   = pd.read_sql_query(
        "SELECT name,emotion,date,time FROM attendance ORDER BY date DESC,time DESC", conn)
    conn.close()
    return df

init_db()

st.sidebar.markdown("## 🎓 AI Attendance")
page = st.sidebar.radio("Navigate", ["📷 Live Monitor", "📊 Attendance Records"])

if page == "📷 Live Monitor":
    st.title("📷 Live Attendance Monitor")
    col1, col2 = st.columns([3, 2])
    with col1:
        img_file = st.camera_input("📸 Take Photo for Attendance")
    with col2:
        st.markdown("### Today Attendance")
        lp = st.empty()

    if img_file is not None:
        try:
            svm_model, label_encoder, emotion_detector = load_models()
            img = Image.open(img_file).convert("RGB")
            img_array = np.array(img)
            gray = np.array(Image.fromarray(img_array).convert("L"))
            resized = np.array(Image.fromarray(gray).resize((48, 48)))
            flat = resized.flatten() / 255.0
            proba = svm_model.predict_proba([flat])[0]
            conf  = max(proba)
            name  = label_encoder.classes_[np.argmax(proba)] if conf > 0.6 else "Unknown"
            emotions = emotion_detector.detect_emotions(img_array)
            emotion = "neutral"
            if emotions:
                emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            if name != "Unknown":
                mark_attendance(name, emotion)
                st.success(f"✅ {name} marked! Emotion: {emotion} ({conf*100:.0f}%)")
            else:
                st.warning("⚠️ Face not recognized!")
            st.image(img, caption=f"{name} - {emotion}", use_container_width=True)
            lp.dataframe(get_attendance(), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.title("📊 Attendance Records")
    df = get_attendance()
    if df.empty:
        st.info("No records yet!")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Total People", df["name"].nunique())
        c3.metric("Top Emotion", df["emotion"].mode()[0])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df["emotion"].value_counts())
        st.download_button("⬇ Download CSV",
                           df.to_csv(index=False).encode(),
                           "attendance.csv")