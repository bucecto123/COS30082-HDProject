import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, embedding_dim=512, index_path="./faiss_index.bin"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = None
        self.labels = [] # To store the names/IDs associated with each embedding

        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim) # L2 for Euclidean distance

    def add_embeddings(self, embeddings, labels):
        # Ensure embeddings are float32 and 2D
        embeddings = np.array(embeddings).astype(np.float32).reshape(-1, self.embedding_dim)
        self.index.add(embeddings)
        self.labels.extend(labels)
        self.save_index()

    def search(self, query_embedding, k=1):
        # Ensure query_embedding is float32 and 2D
        query_embedding = np.array(query_embedding).astype(np.float32).reshape(-1, self.embedding_dim)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            if indices[0][i] != -1: # -1 indicates no match
                results.append({"label": self.labels[indices[0][i]], "distance": distances[0][i]})
            else:
                results.append({"label": "Unknown", "distance": float('inf')})
        return results

    def remove_identity(self, identity_name):
        print(f"Attempting to remove identity: {identity_name}")
        if identity_name not in self.labels:
            print(f"Identity '{identity_name}' not found in the index.")
            return False

        new_embeddings = []
        new_labels = []
        old_embeddings = self.index.reconstruct_n(0, self.index.ntotal) # Get all embeddings

        for i, label in enumerate(self.labels):
            if label != identity_name:
                new_embeddings.append(old_embeddings[i])
                new_labels.append(label)

        if not new_embeddings: # If all identities are removed
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.labels = []
            print(f"All identities removed. Index reset.")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(np.array(new_embeddings).astype(np.float32))
            self.labels = new_labels
            print(f"Identity '{identity_name}' removed. New index size: {self.index.ntotal}")

        self.save_index()
        return True

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        # Save labels separately as FAISS only stores vectors
        np.save(self.index_path + ".labels.npy", np.array(self.labels))

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        self.labels = np.load(self.index_path + ".labels.npy", allow_pickle=True).tolist()
        print(f"FAISS index loaded from {self.index_path} with {self.index.ntotal} embeddings.")