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
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

class AntiSpoofingPredictor:
    def __init__(self, device_id=0):
        self.session = None
        self.input_name = None
        self.output_name = None

    def _load_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, img, model_path):
        self._load_model(model_path)
        model_name = os.path.basename(model_path)
        h_input, w_input = parse_model_name(model_name)
        img = cv2.resize(img, (w_input, h_input))
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        outputs = self.session.run([self.output_name], {self.input_name: img})
        return outputs[0]

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

