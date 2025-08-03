import os
import cv2
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
        # Save original shape for debugging
        orig_shape = face_image.shape
        print(f"[AntiSpoofing] Input shape before processing: {orig_shape}")
        
        # Resize to 80x80 (model's expected input size)
        face_image = cv2.resize(face_image, (80, 80))
        
        # Important: Keep BGR format as the model was trained with BGR
        # Convert to float32 and normalize to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format (channels first)
        face_image = face_image.transpose((2, 0, 1))
        
        # Add batch dimension
        face_image = np.expand_dims(face_image, axis=0)
        
        print(f"[AntiSpoofing] Input shape after processing: {face_image.shape}")

        if self.onnx_session:
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            prediction = self.onnx_session.run([output_name], {input_name: face_image})[0]
            print(f"[AntiSpoofing] ONNX output shape: {prediction.shape}, values: {prediction}")
            
            # The model outputs logits, convert to probabilities
            prediction = prediction[0]  # Remove batch dimension
            
            # Subtract max for numerical stability
            exp_preds = np.exp(prediction - np.max(prediction))
            probabilities = exp_preds / np.sum(exp_preds)
            
            # MiniFASNetV2 class order: [fake, real, background]
            fake_prob = float(probabilities[0])
            real_prob = float(probabilities[1])
            background_prob = float(probabilities[2])
            
            print(f"[AntiSpoofing] Raw logits: {prediction}")
            print(f"[AntiSpoofing] Probabilities: fake={fake_prob:.3f}, real={real_prob:.3f}, background={background_prob:.3f}")
            
            label = np.argmax(probabilities)
            score = real_prob  # Use real face probability
            
            print(f"[AntiSpoofing] Prediction: {'real' if label == 2 else 'fake' if label == 1 else 'background'} face (score: {score:.2f})")
            return score  # Return probability of being real
        else:
            raise RuntimeError("Anti-Spoofing model is not loaded.")
