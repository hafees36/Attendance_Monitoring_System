import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# Path to known faces
KNOWN_FACES_DIR = 'known_faces'
attendance_file = 'attendance.csv'

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png')):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)

# Initialize attendance file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df.to_csv(attendance_file, index=False)

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            # Mark attendance
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            
            # Load existing attendance records
            df = pd.read_csv(attendance_file)
            
            # Avoid duplicate entries for the same person on the same day
            if not ((df['Name'] == name) & (df['Date'] == date)).any():
                new_entry = {'Name': name, 'Date': date, 'Time': time}
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(attendance_file, index=False)
            
            # Display recognized face name
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Attendance System', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

