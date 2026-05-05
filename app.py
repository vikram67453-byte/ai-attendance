import streamlit as st
import cv2
import numpy as np
import pickle
import sqlite3
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
import os

BASE = r"C:\Users\LENOVO"

st.set_page_config(page_title="AI Attendance System", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotion_model = load_model(os.path.join(BASE, "emotion_model.keras"))
    svm_model     = pickle.load(open(os.path.join(BASE, "face_svm.pkl"), "rb"))
    label_encoder = pickle.load(open(os.path.join(BASE, "label_encoder.pkl"), "rb"))
    return face_cascade, emotion_model, svm_model, label_encoder

face_cascade, emotion_model, svm_model, label_encoder = load_models()
EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

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
    col1, col2 = st.columns([3,2])
    with col1:
        run = st.checkbox("▶ Start Camera")
        fp  = st.empty()
    with col2:
        st.markdown("### Today Attendance")
        lp = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                roi     = gray[y:y+h, x:x+w]
                flat    = cv2.resize(roi,(48,48)).flatten()/255.0
                proba   = svm_model.predict_proba([flat])[0]
                conf    = max(proba)
                name    = label_encoder.classes_[np.argmax(proba)] if conf>0.6 else "Unknown"
                color   = (0,255,0) if name!="Unknown" else (0,0,255)
                emo_in  = cv2.resize(roi,(48,48)).reshape(1,48,48,1)/255.0
                emotion = EMOTIONS[np.argmax(emotion_model.predict(emo_in,verbose=0))]
                if name != "Unknown":
                    mark_attendance(name, emotion)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.rectangle(frame,(x,y-50),(x+w,y),color,-1)
                cv2.putText(frame,f"{name} ({conf*100:.0f}%)",(x+5,y-30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.putText(frame,emotion,(x+5,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            fp.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                     channels="RGB", use_container_width=True)
            lp.dataframe(get_attendance(), use_container_width=True, hide_index=True)
        cap.release()

else:
    st.title("📊 Attendance Records")
    df = get_attendance()
    if df.empty:
        st.info("No records yet!")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Total People", df["name"].nunique())
        c3.metric("Top Emotion",  df["emotion"].mode()[0])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df["emotion"].value_counts())
        st.download_button("⬇ Download CSV",
                           df.to_csv(index=False).encode(),
                           "attendance.csv")