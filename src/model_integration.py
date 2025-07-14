import os
import cv2
import math
import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE

# --- Functions extracted from SilentFaceAntiSpoofing/src/utility.py ---
def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale

# --- Classes extracted and simplified from SilentFaceAntiSpoofing/src/data_io/transform.py ---
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array: HWC to CHW, normalize to [0, 1]
            img = torch.from_numpy(pic.transpose((2, 0, 1))).float().div(255)
            return img
        # Add handling for PIL Image if necessary, but for this project, numpy array is expected.
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

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
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        self.model = None
        self.kernel_size = None

    def _load_model(self, model_path):
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, img, model_path):
        self._load_model(model_path)
        model_name = os.path.basename(model_path)
        h_input, w_input, _, _ = parse_model_name(model_name)
        img = cv2.resize(img, (w_input, h_input))
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result

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

