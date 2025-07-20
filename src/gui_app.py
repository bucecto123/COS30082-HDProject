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
    faiss_index = FaissIndex(embedding_dim=128)
    face_system = FaceSystem(faiss_index)
    attendance_logger = AttendanceLogger()
    print("[INFO] System Initialized.")

    # --- PySimpleGUI Layout ---
    sg.theme("LightGrey1")

    left_column = [
        [sg.Text("Camera Feed", size=(40, 1), justification='center')],
        [sg.Image(filename='', key='-IMAGE-', size=(400, 300))]
    ]

    right_column = [
        [sg.Text("Face Recognition Attendance", size=(40, 1), justification='center')],
        [sg.Text("Status: Ready", key='-STATUS-', size=(30, 1))],
        [sg.Button('Register Identity', size=(15, 2)), sg.Button('Recognize Face', size=(15, 2)), sg.Button('Exit', size=(8, 2))],
        [sg.Text("Remove Identity:"), sg.Input(key='-REMOVE_NAME-', size=(15, 1)), sg.Button('Remove Identity', size=(12, 1))],
        [sg.Text("Registered Identities:", size=(30, 1))],
        [sg.Multiline(size=(40, 3), key='-IDENTITIES_LIST-', autoscroll=True, disabled=True, default_text="No identities registered.")],
        [sg.Text("Attendance Log:", size=(30, 1))],
        [sg.Multiline(size=(40, 5), key='-LOG-', autoscroll=True, disabled=True)]
    ]

    layout = [
        [sg.Column(left_column), sg.Column(right_column)]
    ]

    window = sg.Window('Face Recognition App', layout, finalize=True)

    def update_identities_list():
        if face_system.enrolled_identities:
            identities_str = "\n".join(face_system.enrolled_identities.keys())
            window['-IDENTITIES_LIST-'].update(identities_str)
        else:
            window['-IDENTITIES_LIST-'].update("No identities registered.")

    update_identities_list()

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
        display_frame = cv2.resize(frame, (640, 480)) # Resize for display
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
                    update_identities_list()
                else:
                    window['-STATUS-'].update('Status: Registration Failed')

        if event == 'Remove Identity':
            name_to_remove = values['-REMOVE_NAME-']
            if name_to_remove:
                if face_system.remove_identity(name_to_remove):
                    sg.popup('Success', f'Successfully removed {name_to_remove}!')
                    window['-STATUS-'].update(f'Status: Removed {name_to_remove}')
                    update_identities_list()
                else:
                    sg.popup_error('Removal Failed', f'Could not remove {name_to_remove}. Check if the name exists.')
                    window['-STATUS-'].update('Status: Removal Failed')
            else:
                sg.popup_error('Input Error', 'Please enter a name to remove.')

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