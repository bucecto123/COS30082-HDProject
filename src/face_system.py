import os
import cv2
import numpy as np
import torch
import tensorflow as tf
import shutil

from src.model_integration import FaceDetector, AntiSpoofingPredictor, MobileFaceNetEmbeddings
from src.model_integration import parse_model_name
from src.recognition_metrics import euclidean_distance, cosine_similarity, cosine_distance

MOBILEFACENET_MODEL_PATH = "D:/Study/Home_work/COS30082/Project/models/mobilefacenet/MobileFaceNet_9925_9680.pb"
from src.config import ANTISPOOF_THRESHOLD, ANTISPOOF_MODEL_PATH

ANTI_SPOOF_MODEL_PATH = "D:/Study/Home_work/COS30082/Project/models/antispoof/2.7_80x80_MiniFASNetV2.onnx"
CAFFEMODEL_PATH = "D:/Study/Home_work/COS30082/Project/models/mtcnn/Widerface-RetinaFace.caffemodel"
DEPLOY_PROTOTXT_PATH = "D:/Study/Home_work/COS30082/Project/models/mtcnn/deploy.prototxt"

class FaceSystem:
    def __init__(self, faiss_index, device_id=0):
        self.face_detector = FaceDetector(CAFFEMODEL_PATH, DEPLOY_PROTOTXT_PATH)
        self.anti_spoofing_predictor = AntiSpoofingPredictor(device_id)
        self.mobilefacenet_embeddings_model = MobileFaceNetEmbeddings(MOBILEFACENET_MODEL_PATH)
        self.enrolled_identities = {}
        self.faiss_index = faiss_index
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
        # The AntiSpoofingPredictor handles cropping internally
        img_cropped = frame[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
        
        # Use the specific ONNX model path
        prediction = self.anti_spoofing_predictor.predict(img_cropped, ANTI_SPOOF_MODEL_PATH)

        # Apply softmax to get probabilities
        softmax_scores = torch.nn.functional.softmax(torch.from_numpy(prediction), dim=1).numpy()
        # The model outputs [fake, photo, real]
        real_score = softmax_scores[0][2]

        print(f"Anti-spoofing real score: {real_score:.4f}, Threshold: {ANTISPOOF_THRESHOLD}")

        if real_score >= ANTISPOOF_THRESHOLD:  # Real Face
            # 3. Face Embedding (MobileFaceNet)
            face_img = frame[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
            face_img = cv2.resize(face_img, (112, 112))
            embedding = self.mobilefacenet_embeddings_model.get_embeddings(face_img)
            
            return embedding, f"Real Face, Score: {real_score:.2f}"
        else:  # Fake Face
            return None, f"Fake Face, Score: {real_score:.2f}"

    def remove_identity(self, identity_name):
        # Remove from FAISS index
        if self.faiss_index.remove_identity(identity_name):
            # Remove from enrolled_identities dictionary
            if identity_name in self.enrolled_identities:
                del self.enrolled_identities[identity_name]
                self.save_identities()

            # Remove face images directory
            face_dir = os.path.join("data", "faces", identity_name)
            if os.path.exists(face_dir):
                shutil.rmtree(face_dir)
                print(f"Removed face image directory for {identity_name}")
            return True
        return False

if __name__ == '__main__':
    # Example Usage (requires a test image)
    # You'll need to replace 'path/to/your/image.jpg' with an actual image path
    # and ensure the model paths are correct.

    
    test_image_path = "D:/Study/Home_work/COS30082/Project/SilentFaceAntiSpoofing/images/sample/image_F1.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        print("Please update 'test_image_path' in face_system.py to a valid image.")
    else:
        # Initialize FaissIndex for testing purposes
        from src.verification.classifier.faiss_index import FaissIndex
        faiss_index_test = FaissIndex(embedding_dim=128) # Assuming 128 is the embedding dimension
        face_system = FaceSystem(faiss_index_test)
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
                # faiss_index = FaissIndex()
                # distances, indices = faiss_index.search(embedding, k=1)
                # if distances[0][0] < threshold:
                #     print(f"Recognized as user with ID: {indices[0][0]}")
                # else:
                #     print("User not recognized")