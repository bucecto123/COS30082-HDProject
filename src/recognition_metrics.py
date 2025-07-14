import numpy as np
from numpy.linalg import norm

def euclidean_distance(embedding1, embedding2):
    return norm(embedding1 - embedding2)

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = norm(embedding1)
    norm_embedding2 = norm(embedding2)
    if norm_embedding1 == 0 or norm_embedding2 == 0:
        return 0.0  # Handle zero-norm case to avoid division by zero
    return dot_product / (norm_embedding1 * norm_embedding2)

def cosine_distance(embedding1, embedding2):
    return 1 - cosine_similarity(embedding1, embedding2)
