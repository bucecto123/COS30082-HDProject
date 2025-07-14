
import os
import cv2
import numpy as np
import torch
import tensorflow as tf

from src.model_integration import FaceDetector, AntiSpoofingPredictor, MobileFaceNetEmbeddings
from src.model_integration import parse_model_name
from src.recognition_metrics import euclidean_distance, cosine_similarity, cosine_distance

MOBILEFACENET_MODEL_PATH = "D:/Study/Home_work/COS30082/Project/models/mobilefacenet/MobileFaceNet_9925_9680.pb"
ANTI_SPOOF_MODEL_DIR = "D:/Study/Home_work/COS30082/Project/models/antispoof/"
CAFFEMODEL_PATH = "D:/Study/Home_work/COS30082/Project/models/mtcnn/Widerface-RetinaFace.caffemodel"
DEPLOY_PROTOTXT_PATH = "D:/Study/Home_work/COS30082/Project/models/mtcnn/deploy.prototxt"

class FaceSystem:
    def __init__(self, device_id=0):
        self.face_detector = FaceDetector(CAFFEMODEL_PATH, DEPLOY_PROTOTXT_PATH)
        self.anti_spoofing_predictor = AntiSpoofingPredictor(device_id)
        self.mobilefacenet_embeddings_model = MobileFaceNetEmbeddings(MOBILEFACENET_MODEL_PATH)
        self.enrolled_identities = {}
        self.faiss_index = None # Placeholder for FAISS index
        self.identities_file = "./data/enrolled_identities.pkl"
        self.load_identities()

    def enroll_identity(self, name, embedding):
        self.enrolled_identities[name] = embedding
        self.save_identities()
        print(f"Enrolled {name} with embedding shape {embedding.shape}")

    def save_identities(self):
        import pickle
        with open(self.identities_file, 'wb') as f:
            pickle.dump(self.enrolled_identities, f)
        print(f"Enrolled identities saved to {self.identities_file}")

    def load_identities(self):
        import pickle
        if os.path.exists(self.identities_file):
            with open(self.identities_file, 'rb') as f:
                self.enrolled_identities = pickle.load(f)
            print(f"Enrolled identities loaded from {self.identities_file}")
        else:
            print("No enrolled identities file found. Starting with empty identities.")

    def recognize_face(self, query_embedding, threshold_euclidean=0.8, threshold_cosine=0.5):
        best_match_euclidean = {"name": "Unknown", "distance": float('inf')}
        best_match_cosine = {"name": "Unknown", "similarity": -1.0}

        for name, enrolled_embedding in self.enrolled_identities.items():
            # Euclidean Distance
            dist_euclidean = euclidean_distance(query_embedding, enrolled_embedding)
            if dist_euclidean < best_match_euclidean["distance"]:
                best_match_euclidean["distance"] = dist_euclidean
                best_match_euclidean["name"] = name

            # Cosine Similarity
            sim_cosine = cosine_similarity(query_embedding, enrolled_embedding)
            if sim_cosine > best_match_cosine["similarity"]:
                best_match_cosine["similarity"] = sim_cosine
                best_match_cosine["name"] = name

        recognized_name = "Unknown"
        recognition_method = "None"

        if best_match_euclidean["distance"] < threshold_euclidean:
            recognized_name = best_match_euclidean["name"]
            recognition_method = f"Euclidean (Dist: {best_match_euclidean['distance']:.2f})"
        
        if best_match_cosine["similarity"] > threshold_cosine:
            if recognized_name == "Unknown" or recognized_name == best_match_cosine["name"]:
                recognized_name = best_match_cosine["name"]
                recognition_method = f"Cosine (Sim: {best_match_cosine['similarity']:.2f})"

        return recognized_name, recognition_method

    def process_frame(self, frame):
        # 1. Face Detection
        image_bbox = self.face_detector.get_bbox(frame)
        if not image_bbox:
            return None, "No face detected"

        # 2. Anti-Spoofing
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(ANTI_SPOOF_MODEL_DIR):
            if not model_name.endswith('.pth'):
                continue
            model_path = os.path.join(ANTI_SPOOF_MODEL_DIR, model_name)
            # The AntiSpoofingPredictor handles cropping internally now
            img_cropped = frame[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
            prediction += self.anti_spoofing_predictor.predict(img_cropped, model_path)

        label = np.argmax(prediction)
        value = prediction[0][label] / 2

        if label == 1:  # Real Face
            # 3. Face Embedding (MobileFaceNet)
            face_img = frame[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
            face_img = cv2.resize(face_img, (112, 112)) # RESIZE TO 112x112 for MobileFaceNet
            embedding = self.mobilefacenet_embeddings_model.get_embeddings(face_img)
            
            return embedding, "Real Face, Score: {:.2f}".format(value)
        else:  # Fake Face
            return None, "Fake Face, Score: {:.2f}".format(value)

if __name__ == '__main__':
    # Example Usage (requires a test image)
    # You'll need to replace 'path/to/your/image.jpg' with an actual image path
    # and ensure the model paths are correct.
    
    # For testing, you can use an image from the Silent-Face-Anti-Spoofing/images/sample directory
    # For example: D:/Study/Home_work/COS30082/Project/Silent-Face-Anti-Spoofing/images/sample/image_F1.jpg
    
    test_image_path = "D:/Study/Home_work/COS30082/Project/SilentFaceAntiSpoofing/images/sample/image_F1.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        print("Please update 'test_image_path' in face_system.py to a valid image.")
    else:
        face_system = FaceSystem()
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Error: Could not read image from {test_image_path}")
        else:
            embedding, status = face_system.process_frame(image)
            print(f"Status: {status}")
            if embedding is not None:
                print(f"Embedding shape: {embedding.shape}")
                # In a real system, you would now compare this embedding to your database
                # For example:
                # from src.verification.classifier.faiss_index import FaissIndex
                # faiss_index = FaissIndex("path/to/your/faiss_index.bin")
                # distances, indices = faiss_index.search(embedding, k=1)
                # if distances[0][0] < threshold:
                #     print(f"Recognized as user with ID: {indices[0][0]}")
                # else:
                #     print("User not recognized")
