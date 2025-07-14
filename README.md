# Real-time Facial Recognition Attendance System for Jetson

This project implements a real-time facial recognition attendance system optimized for NVIDIA Jetson devices. It leverages MobileFaceNet for face recognition, MTCNN for face detection, includes an anti-spoofing mechanism, utilizes a FAISS vector database for efficient similarity search, and is optimized with TensorRT for improved performance.

## Features

-   **Real-time Face Detection:** Uses MTCNN to accurately detect faces in video streams.
-   **Face Recognition:** Employs MobileFaceNet for robust facial feature extraction and recognition.
-   **Anti-Spoofing:** Incorporates a basic anti-spoofing mechanism to prevent fraudulent attendance.
-   **FAISS Vector Database:** Efficiently stores and searches facial embeddings for quick recognition.
-   **TensorRT Optimization:** Accelerates model inference for MobileFaceNet and Anti-Spoofing models on Jetson.
-   **Multi-threading:** Processes video frames and performs recognition in parallel for enhanced performance.
-   **Student Enrollment Interface:** A simple command-line interface to enroll new students by capturing their facial data.
-   **Performance Monitoring:** Displays real-time FPS to monitor system performance.

## Project Structure

```
D:/Study/Home_work/COS30082/Project/
├───data/
│   ├───faces/              # Stores enrolled student face images
│   └───attendance.csv      # Records attendance logs
├───models/
│   ├───mtcnn/              # MTCNN model files
│   ├───mobilefacenet/      # MobileFaceNet model files (.h5 or .trt)
│   └───antispoof/          # Anti-spoofing model files (.onnx or .trt)
├───src/
│   ├───antispoof/
│   │   └───anti_spoofing.py    # Anti-spoofing model and logic
│   ├───attendance/
│   ├───detection/
│   ├───utils/
│   │   └───tensorrt_utils.py   # TensorRT inference utility
│   ├───verification/
│   │   ├───classifier/
│   │   │   └───faiss_index.py  # FAISS index building and management
│   │   └───mobilefacenet.py    # MobileFaceNet model and embedding logic
│   ├───gui_app.py          # Main real-time attendance system GUI application
│   ├───face_system.py      # Core facial recognition and attendance logic
│   └───config.py           # Configuration parameters
└───requirements.txt        # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Project
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Note:** `tensorflow`, `faiss-cpu`, `mtcnn`, `opencv-python`, `torch`, `torchvision`, `onnxruntime`, and `pandas` might require specific installations depending on your Jetson environment and CUDA/cuDNN versions. Refer to their official documentation for Jetson-specific installation guides.

### 3. Download and Convert Pre-trained Models

This project assumes you have pre-trained models for MTCNN, MobileFaceNet, and an Anti-Spoofing model. You need to place these models in their respective directories:

-   **MTCNN:** Place MTCNN model files (e.g., `pnet.npy`, `rnet.npy`, `onet.npy`) into `models/mtcnn/`.
-   **MobileFaceNet:** Place your pre-trained MobileFaceNet model (e.g., `mobilefacenet.h5`) into `models/mobilefacenet/`. If you have a TensorRT engine, place `mobilefacenet.trt` here.
-   **Anti-Spoofing (Silent-Face-Anti-Spoofing):** The models from the [Silent-Face-Anti-Spoofing repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) are typically in `.pth` (PyTorch) format. **These `.pth` models are not directly compatible with this system.** You need to convert them to ONNX format first, and then place the converted `.onnx` file (e.g., `2.7_80x80_MiniFASNetV2.onnx` or `4_0_0_80x80_MiniFASNetV1SE.onnx`) into `models/antispoof/`. If you have converted it further to a TensorRT engine, place `antispoofing_model.trt` here.

    **Converting .pth to ONNX:**
    Refer to the Silent-Face-Anti-Spoofing repository's documentation or common PyTorch to ONNX export methods. A typical approach involves loading the `.pth` model in PyTorch and then using `torch.onnx.export`.

**TensorRT Conversion (Optional but Recommended for Jetson):**

To leverage TensorRT for optimized inference, you will need to convert your TensorFlow/Keras models (`.h5`) or ONNX models (`.onnx`) to TensorRT engines (`.trt`). This process is specific to your Jetson device and TensorRT installation. You can typically use `tf.saved_model.save` to save your Keras model as a SavedModel, and then use `tf.experimental.tensorrt.Converter` or `trtexec` to convert it to a `.trt` engine. For ONNX models, you can use `trtexec` directly or the TensorRT Python API.

### 4. Configure the System

Edit `src/config.py` to adjust paths, thresholds, and camera index as needed.

## Usage

### Run the Attendance System (GUI)

Launch the main GUI application. This application handles both student enrollment and real-time attendance.

```bash
python src/gui_app.py
```

**Enrollment:** Within the GUI, use the "Register Identity" button to enroll new students. The system will guide you through capturing facial data and automatically update the FAISS index.

**Attendance:** The system will continuously detect and recognize faces from the camera feed, performing anti-spoofing checks and logging attendance for recognized individuals.

The system will open a webcam feed, detect faces, recognize enrolled students, perform anti-spoofing checks, and mark attendance in `data/attendance.csv`.

Press 'q' to quit the application.
