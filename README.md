# Face Recognition Based Attendance System

This project is a real-time attendance system that uses face recognition to identify individuals and log their attendance. It features a graphical user interface (GUI) for easy interaction, and is designed to be deployable on edge devices like the NVIDIA Jetson series.

## Features

*   **Real-time Face Recognition:** Identifies users from a live video stream.
*   **Attendance Logging:** Logs attendance to a CSV file with timestamps.
*   **User Management:**
    *   Enroll new users by capturing their face images.
    *   Remove existing users from the system.
*   **Liveness Detection:** Prevents spoofing attacks using a liveness detection model.
*   **GUI:** An easy-to-use interface for managing the system.

## Models Used

This project utilizes pre-trained models from the following repositories:

*   **MobileFaceNet:** For face recognition, we use the MobileFaceNet model from [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF). This model is lightweight and efficient, making it ideal for edge devices.
*   **Silent-Face-Anti-Spoofing:** For liveness detection, we use the models from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing). These models are effective at preventing spoofing attacks from photos and videos.

We are grateful to the authors of these repositories for making their work publicly available.

## Deployment on Jetson

To deploy this project on a Jetson device, you will need to make the following adjustments:

1.  **Install System Dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
    ```

2.  **Install PyTorch and Torchvision:**
    Follow the instructions on the NVIDIA forums to install the correct versions of PyTorch and Torchvision for your Jetson device.

3.  **Install ONNX Runtime for GPU:**
    ```bash
    pip install onnxruntime-gpu
    ```

4.  **Install FAISS for GPU:**
    You will need to build and install FAISS from source with GPU support.

5.  **Convert Models to TensorRT Engines:**
    To get the best performance, you should convert the ONNX models to TensorRT engines using the provided utilities in `src/utils/tensorrt_utils.py`.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python src/gui_app.py
    ```