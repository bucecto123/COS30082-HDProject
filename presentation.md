
# Presentation: Real-Time Face Recognition Attendance System

### 1. Introduction & Project Goal

*   **What is it?** This project is a comprehensive, real-time attendance system that automates the process of tracking presence by using facial recognition.
*   **Problem It Solves:** It replaces manual attendance taking, which is time-consuming and prone to errors. It also enhances security by preventing "buddy punching" (where someone clocks in for an absent colleague) through advanced liveness detection.
*   **Core Objective:** To create a secure, efficient, and user-friendly system that can be deployed on standard computers and is optimized for low-power edge devices like the NVIDIA Jetson.

---

### 2. System Architecture & Workflow

The system operates as a multi-stage pipeline. Each frame from the video camera is processed through these stages in real-time:

1.  **Video Capture (Input):**
    *   The system captures video from a connected camera using **OpenCV**.

2.  **Face Detection (Where is the face?):**
    *   It uses a **MTCNN (Multi-task Cascaded Convolutional Networks)** model to scan the frame and identify the precise location and bounding box of any human faces.

3.  **Liveness Detection / Anti-Spoofing (Is it a real person?):**
    *   For each detected face, the system performs a critical anti-spoofing check.
    *   It uses a **MiniFASNet (Mini Face Anti-Spoofing Network)** model to analyze the face texture and micro-movements to determine if it's a live person in front of the camera or a 2D image (like a photo or a video on a phone).
    *   This is a crucial security layer. If the check fails, the process stops for that face.

4.  **Face Recognition (Who is this person?):**
    *   If the face is determined to be "live," it's processed by the **MobileFaceNet** model.
    *   MobileFaceNet is a highly efficient deep learning model that converts the face into a unique numerical vector called an **embedding** (a list of 128 numbers that represents the facial features).

5.  **Identity Verification (Do we know this person?):**
    *   This new embedding is then compared against a database of pre-enrolled user embeddings.
    *   The system uses **FAISS (Facebook AI Similarity Search)**, a library designed for extremely fast similarity searches among millions of vectors. FAISS compares the new embedding to all the known embeddings in its index (`faiss_index.bin`).
    *   If a close match is found, the system identifies the user.

6.  **Attendance Logging (Mark them present):**
    *   Once a registered user is identified, the system logs their name and the current timestamp into an **`attendance.csv`** file. The system is designed to log a person's attendance only once per session to avoid duplicate entries.

7.  **User Interface (Control Center):**
    *   The entire process is managed through a **GUI (Graphical User Interface)** built in Python. The GUI allows an administrator to:
        *   View the live camera feed with recognized faces and liveness status overlaid.
        *   **Enroll New Users:** Capture their face, generate an embedding, and add it to the FAISS index.
        *   **Remove Users:** Delete a user's data and embedding from the system.
        *   View a list of all enrolled identities.

---

### 3. Implementation Details & Key Technologies

*   **Programming Language:** Python
*   **Core Libraries:**
    *   **OpenCV:** For all image and video processing tasks.
    *   **TensorFlow/PyTorch:** Used for the original model creation and training.
    *   **ONNX Runtime:** Used for running the inference on the standardized `.onnx` models.
    *   **FAISS:** For high-speed similarity search in the face recognition step.
    *   **Pandas:** For managing and writing to the `attendance.csv` log.
*   **Models & Acknowledgements:**
    *   **Face Detection:** MTCNN (`deploy.prototxt`, `Widerface-RetinaFace.caffemodel`)
    *   **Liveness Detection:** MiniFASNetV1SE & MiniFASNetV2 (`.onnx` models). These models are from the [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) project by minivision-ai.
    *   **Face Recognition:** MobileFaceNet (`.pb` converted to `.onnx`). This model is from the [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF) project by sirius-ai.
    *   We are grateful to the creators of these repositories for their open-source contributions.

---

### 4. Preparation for Jetson Deployment (Edge AI)

A key goal was to ensure this system can run efficiently on an NVIDIA Jetson, which is a small, powerful computer for AI applications at the edge. Several steps were taken to prepare for this:

1.  **Model Standardization to ONNX:**
    *   **Problem:** Different models were in different formats (TensorFlow `.pb`, Caffe `.caffemodel`, PyTorch `.pth`). These are not always portable.
    *   **Solution:** All core models for inference (Anti-Spoofing, Face Recognition) have been converted to the **ONNX (Open Neural Network Exchange)** format. This is a universal format that can run on many different types of hardware.
    *   **Action Taken:** I ran the `convert_mobilefacenet_to_onnx.py` script to create `mobilefacenet.onnx` and have the other `.onnx` models ready.

2.  **Planning for Hardware Acceleration with TensorRT:**
    *   **Problem:** To get maximum performance on NVIDIA GPUs (like the one in a Jetson), you need to use **TensorRT**. TensorRT optimizes the ONNX model for the specific GPU, resulting in significantly lower latency and higher throughput.
    *   **Solution:** I created the `convert_to_tensorrt.py` script. This script is designed to be run **on the Jetson device** to convert the `.onnx` files into highly optimized `.engine` files.
    *   **Why not run it now?** TensorRT engines are specific to the hardware they're built on. An engine built on your Windows PC's GPU will not work on a Jetson. The script is ready for immediate use once the project is moved to the Jetson.

3.  **Creating a Jetson-Ready Environment:**
    *   **Problem:** The software dependencies for a standard PC are different from a Jetson (which uses an ARM-based CPU).
    *   **Solution:** I have updated the `README.md` and `requirements.txt` with specific instructions for a Jetson deployment. This includes:
        *   Installing GPU-specific versions of libraries like `onnxruntime-gpu` and `faiss-gpu`.
        *   Providing instructions on where to get the correct PyTorch build for Jetson.
        *   Commenting out the CPU-only packages in `requirements.txt` to avoid conflicts.

By taking these steps, the project is not just a prototype; it is architected and prepared for a seamless transition to a high-performance, production-ready state on an edge device.
