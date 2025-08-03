import os
import cv2
import math
import numpy as np
import tensorflow as tf
import onnxruntime

# --- Functions extracted and simplified from SilentFaceAntiSpoofing/src/utility.py ---
def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    return int(h_input), int(w_input)

class FaceDetector:
    def __init__(self, caffemodel_path, deploy_prototxt_path):
        self.detector = cv2.dnn.readNetFromCaffe(deploy_prototxt_path, caffemodel_path)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        confidence = out[max_conf_index, 2]
        if confidence > self.detector_confidence:
            left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                       out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
            bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
            return bbox
        return None

class AntiSpoofingPredictor:
    def __init__(self, device_id=0):
        # Keep one ONNX session alive to avoid heavy reload each inference
        self.session = None
        self.input_name = None
        self.output_name = None
        self.current_model_path = None  # track which model is currently loaded

    def _load_model(self, model_path):
        """Load the ONNX model only if it is not already loaded."""
        if self.session is None or model_path != self.current_model_path:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=onnxruntime.get_available_providers()
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.current_model_path = model_path

    def predict(self, img, model_path):
        # Ensure model is loaded (will be a no-op if already loaded)
        self._load_model(model_path)

        model_name = os.path.basename(model_path)
        h_input, w_input = parse_model_name(model_name)

        # Pre-processing -----------------------------------------------------
        # Save original shape for debugging
        orig_shape = img.shape
        print(f"[AntiSpoofing] Original input shape: {orig_shape}")
        
        # 1. Resize to model input (80x80)
        img = cv2.resize(img, (w_input, h_input))
        
        # 2. Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 3. Convert HWC to CHW format
        img = img.transpose((2, 0, 1))
        
        # 4. Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        print(f"[AntiSpoofing] Processed input shape: {img.shape}")
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img})
        prediction = outputs[0]
        
        print(f"[AntiSpoofing] Raw output shape: {prediction.shape}")
        print(f"[AntiSpoofing] Raw output values: {prediction}")
        
        # Apply softmax to get probabilities
        prediction = prediction[0]  # Remove batch dimension
        exp_preds = np.exp(prediction - np.max(prediction))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # MiniFASNetV2 class order: [background, fake, real]
        background_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])
        real_prob = float(probabilities[2])
        
        print(f"[AntiSpoofing] Probabilities: background={background_prob:.3f}, fake={fake_prob:.3f}, real={real_prob:.3f}")
        
        score = real_prob  # Use probability of real class
        return score

class MobileFaceNetEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self._load_model()

    def _load_model(self):
        with self.graph.as_default():
            with tf.io.gfile.GFile(self.model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.input_tensor = self.graph.get_tensor_by_name("input:0")
        self.embeddings_tensor = self.graph.get_tensor_by_name("embeddings:0")

    def get_embeddings(self, image):
        # Preprocess image: normalize to [-1, 1]
        image = image.astype(np.float32)
        image = (image - 127.5) * 0.0078125
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        feed_dict = {self.input_tensor: image}
        embeddings = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)
        return embeddings

