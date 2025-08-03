import os
import numpy as np
import onnxruntime
from src.config import ANTISPOOF_MODEL_PATH, INPUT_SIZE

class AntiSpoofing:
    def __init__(self):
        self.onnx_session = None
        self._load_model()

    def _load_model(self):
        onnx_path = os.path.join(ANTISPOOF_MODEL_PATH, "2.7_80x80_MiniFASNetV2.onnx")

        if os.path.exists(onnx_path):
            print(f"Loading Anti-Spoofing ONNX model from {onnx_path}")
            self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        else:
            raise FileNotFoundError(f"Anti-Spoofing model not found at {onnx_path}. Please convert your .pth model to ONNX or TensorRT.")

    def predict(self, face_image):
        # Ensure the image is in the correct format (batch, height, width, channels)
        if face_image.ndim == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Preprocess the image (e.g., normalize pixel values)
        face_image = (face_image.astype(np.float32) / 255.0 - 0.5) * 2

        if self.onnx_session:
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            prediction = self.onnx_session.run([output_name], {input_name: face_image})[0]
            print(f"[AntiSpoofing] ONNX output shape: {prediction.shape}, values: {prediction}")
            return prediction[0] # Return the full prediction array
        else:
            raise RuntimeError("Anti-Spoofing model is not loaded.")
