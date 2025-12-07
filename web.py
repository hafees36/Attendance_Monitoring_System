from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# === Directories and Files ===
KNOWN_FACES_DIR = 'known_faces'
attendance_file = 'webattendance.csv'

# === Load Known Faces ===
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png')):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{filename}')
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"⚠ No face found in {filename}, skipping.")

# === Initialize Attendance File ===
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
    df.to_csv(attendance_file, index=False)

# === Start Webcam ===
video_capture = cv2.VideoCapture(0)

# === Function to Mark Absent Students ===
def mark_absentees(known_names, attendance_file):
    if not os.path.exists(attendance_file):
        return
    
    df = pd.read_csv(attendance_file)
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Names already marked present today
    present_today = df[df['Date'] == today]['Name'].tolist()

    # Who’s missing
    absentees = [name for name in known_names if name not in present_today]

    for name in absentees:
        new_entry = {'Name': name, 'Date': today, 'Time': '-', 'Status': 'Absent'}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df.to_csv(attendance_file, index=False)
    print(f"📋 Marked absentees for {today}: {absentees}")

# === Frame Generator ===
def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                time = now.strftime('%H:%M:%S')

                df = pd.read_csv(attendance_file)
                if not ((df['Name'] == name) & (df['Date'] == date)).any():
                    new_entry = {'Name': name, 'Date': date, 'Time': time, 'Status': 'Present'}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv(attendance_file, index=False)

                # Draw rectangle & label
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame for web display
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === Flask Routes ===
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    df = pd.read_csv(attendance_file)
    attendance_records = df.to_dict(orient='records')
    return render_template('index.html', attendance=attendance_records)

@app.route('/attendance_data')
def attendance_data():
    df = pd.read_csv(attendance_file)
    return jsonify(df.to_dict(orient='records'))

# === Run the App ===
if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        # When the app is stopped, mark absentees automatically
        mark_absentees(known_face_names, attendance_file)
        video_capture.release()