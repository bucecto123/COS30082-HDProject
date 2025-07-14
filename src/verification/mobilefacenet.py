import tensorflow as tf
import numpy as np
import os
from src.config import MOBILEFACENET_MODEL_PATH, INPUT_SIZE

class MobileFaceNet:
    def __init__(self):
        self.trt_engine = None
        self.tf_model = None
        self._load_model()

    def _load_model(self):
        trt_path = os.path.join(MOBILEFACENET_MODEL_PATH, "mobilefacenet.trt")
        if os.path.exists(trt_path):
            from src.utils.tensorrt_utils import TRTInference
            print(f"Loading MobileFaceNet TensorRT engine from {trt_path}")
            self.trt_engine = TRTInference(trt_path)
        else:
            print(f"MobileFaceNet TensorRT engine not found at {trt_path}. Loading TensorFlow model.")
            model_path = os.path.join(MOBILEFACENET_MODEL_PATH, "mobilefacenet.h5")
            if os.path.exists(model_path):
                print(f"Loading MobileFaceNet TensorFlow model from {model_path}")
                self.tf_model = tf.keras.models.load_model(model_path)
            else:
                print(f"MobileFaceNet TensorFlow model not found at {model_path}. Creating a dummy model.")
                self.tf_model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128) # Output embedding size
                ])

    def embed(self, face_image):
        # Ensure the image is in the correct format (batch, height, width, channels)
        if face_image.ndim == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Preprocess the image (e.g., normalize pixel values)
        face_image = face_image / 255.0 # Normalize to [0, 1]

        if self.trt_engine:
            embedding = self.trt_engine.infer(face_image)
        else:
            embedding = self.tf_model.predict(face_image)
        return embedding