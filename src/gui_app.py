import sys
sys.path.append("D:/Study/Home_work/COS30082/Project")

import PySimpleGUI as sg
import cv2
import numpy as np
import time

from src.face_system import FaceSystem
from src.verification.classifier.faiss_index import FaissIndex
from src.attendance.attendance_logger import AttendanceLogger

def main():
    # --- System Initialization ---
    print("[INFO] Initializing FaceSystem...")
    face_system = FaceSystem()
    faiss_index = FaissIndex(embedding_dim=128)
    attendance_logger = AttendanceLogger()
    print("[INFO] System Initialized.")

    # --- PySimpleGUI Layout ---
    sg.theme("LightGrey1")

    layout = [
        [sg.Text("Face Recognition Attendance", size=(60, 1), justification='center')],
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Text("Status: Ready", key='-STATUS-', size=(40, 1))],
        [sg.Button('Register Identity', size=(20, 2)), sg.Button('Recognize Face', size=(20, 2)), sg.Button('Exit', size=(10, 2))],
        [sg.Text("Attendance Log:", size=(40, 1))],
        [sg.Multiline(size=(60, 10), key='-LOG-', autoscroll=True, disabled=True)]
    ]

    window = sg.Window('Face Recognition App', layout)

    # --- Video Capture ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sg.popup_error("Camera Error", "Unable to open video source")
        return

    # --- Event Loop ---
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()
        if not ret:
            sg.popup_error("Frame Error", "Failed to capture frame from camera.")
            break

        # --- Face Detection for visual feedback ---
        image_bbox = face_system.face_detector.get_bbox(frame)
        if image_bbox:
            x, y, w, h = image_bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # --- Resize and Update GUI Image ---
        display_frame = cv2.resize(frame, (800, 600)) # Resize for display
        imgbytes = cv2.imencode('.png', display_frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

        # --- Button Events (use original 'frame' for quality) ---
        if event == 'Register Identity':
            name = sg.popup_get_text('Enter your name:', title='Register Identity')
            if name:
                embeddings = []
                for i in range(5):
                    window['-STATUS-'].update(f'Status: Capturing image {i+1}/5 for {name}...')
                    window.refresh()
                    time.sleep(1) # Give user time to position their face
                    ret, frame = cap.read()
                    if not ret:
                        sg.popup_error("Frame Error", "Failed to capture frame from camera.")
                        break
                    embedding, status = face_system.process_frame(frame)
                    if embedding is not None and "Real Face" in status:
                        embeddings.append(embedding.flatten())
                        print(f"Captured image {i+1}/5 for {name}")
                    else:
                        sg.popup_error('Registration Failed', f'Could not capture image {i+1}/5: {status}')
                
                if len(embeddings) == 5:
                    avg_embedding = np.mean(embeddings, axis=0)
                    face_system.enroll_identity(name, avg_embedding)
                    faiss_index.add_embeddings([avg_embedding], [name])
                    sg.popup('Success', f'Successfully registered {name}!')
                    window['-STATUS-'].update(f'Status: Registered {name}')
                else:
                    window['-STATUS-'].update('Status: Registration Failed')

        if event == 'Recognize Face':
            window['-STATUS-'].update('Status: Recognizing face...')
            window.refresh()
            embedding, status = face_system.process_frame(frame)
            if embedding is not None and "Real Face" in status:
                recognized_name, method = face_system.recognize_face(embedding.flatten())
                if recognized_name != "Unknown":
                    attendance_logger.log_attendance(recognized_name, "Present", method)
                    window['-LOG-'].print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {recognized_name} - Present ({method})")
                sg.popup('Recognition Result', f'Recognized: {recognized_name} ({method})')
                window['-STATUS-'].update(f'Status: Recognized {recognized_name}')
            else:
                sg.popup_error('Recognition Failed', f'Could not recognize: {status}')
                window['-STATUS-'].update('Status: Recognition Failed')

    cap.release()
    window.close()

if __name__ == '__main__':
    main()