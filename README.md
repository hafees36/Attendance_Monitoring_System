# Face Recognition Attendance System (Flask + OpenCV)

This is a simple **web-based attendance system** that uses **face recognition** to mark students **Present** or **Absent** automatically.

- Recognizes faces from a webcam feed using [`face_recognition`](https://github.com/ageitgey/face_recognition)
- Marks **Present** in a CSV when a known face is detected
- Automatically marks **Absent** for all remaining students when the app stops
- Displays live attendance data on a web page using **Flask**
- You should make a folder named known_faces to input the training dataset
---

## 🧩 How It Works

1. **Known faces** are stored as images in the `known_faces/` folder.
   - The filename (without extension) is used as the **student name**.
   - Example: `known_faces/Alice.jpg` → `Alice`

2. At startup, the app:
   - Encodes all faces in `known_faces/`
   - Ensures `webattendance.csv` exists with columns:
     - `Name`, `Date`, `Time`, `Status`

3. While running:
   - Captures frames from the webcam (`cv2.VideoCapture(0)`)
   - Detects and recognizes faces
   - If a known face is found:
     - Marks them as **Present** in `webattendance.csv`
     - Only once per day

4. When the app is stopped:
   - For all known names **not marked Present today**, it adds entries with:
     - `Status = 'Absent'` and `Time = '-'`

---

## 🛠 Tech Stack

- **Python**
- **Flask**
- **OpenCV** (`cv2`)
- **face_recognition**
- **NumPy**
- **Pandas**
- **HTML/Jinja2** (for `index.html` template)

---

## 📁 Project Structure

A typical structure for this project:

```bash
project/
├── app.py                # Main Flask application (this file)
├── known_faces/          # Folder containing known face images
│   ├── Alice.jpg
│   ├── Bob.png
│   └── ...
├── templates/
│   └── index.html        # Frontend to show attendance & video feed
├── webattendance.csv     # Attendance log (auto-created)
└── requirements.txt      # Python dependencies
