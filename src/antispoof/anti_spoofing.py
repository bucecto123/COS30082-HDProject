import tensorflow as tf
import numpy as np
import os
import onnxruntime
from src.config import ANTISPOOF_MODEL_PATH, INPUT_SIZE

class AntiSpoofing:
    def __init__(self):
        self.onnx_session = None
        self.trt_engine = None
        self._load_model()

    def _load_model(self):
        onnx_path = os.path.join(ANTISPOOF_MODEL_PATH, "Silent-Face-Anti-Spoofing.onnx")
        trt_path = os.path.join(ANTISPOOF_MODEL_PATH, "antispoofing_model.trt")

        if os.path.exists(trt_path):
            from src.utils.tensorrt_utils import TRTInference
            print(f"Loading Anti-Spoofing TensorRT engine from {trt_path}")
            self.trt_engine = TRTInference(trt_path)
        elif os.path.exists(onnx_path):
            print(f"Loading Anti-Spoofing ONNX model from {onnx_path}")
            self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        else:
            print(f"Anti-Spoofing model not found at {onnx_path} or {trt_path}. Please convert your .pth model to ONNX or TensorRT.")
            # Create a dummy model if no valid model is found
            self.tf_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid') # Output a probability for liveness
            ])

    def predict(self, face_image):
        # Ensure the image is in the correct format (batch, height, width, channels)
        if face_image.ndim == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Preprocess the image (e.g., normalize pixel values)
        face_image = face_image.astype(np.float32) / 255.0 # Normalize to [0, 1]

        if self.onnx_session:
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            prediction = self.onnx_session.run([output_name], {input_name: face_image})[0]
        elif self.trt_engine:
            prediction = self.trt_engine.infer(face_image)
        else:
            # Fallback to dummy TF model if no ONNX/TRT model is loaded
            prediction = self.tf_model.predict(face_image)
        return prediction[0][0] # Return the probability of being live