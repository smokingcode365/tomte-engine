# tomte/engine/dictionary/logic.py

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List

def calculate_topic_centroids(vectors: List[List[float]], eps: float = 0.5, min_samples: int = 2) -> List[List[float]]:
    """
    Uses DBSCAN to group vectors and return centroids for Phase 1.
    """
    if not vectors: return []
    
    X = np.array(vectors)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
    
    centroids = []
    for label in set(clustering.labels_):
        if label == -1: continue # Ignore noise
        mask = clustering.labels_ == label
        centroids.append(np.mean(X[mask], axis=0).tolist())
        
    return centroids if centroids else [np.mean(X, axis=0).tolist()]

def nudge_vector(old_vector: List[float], new_sample: List[float], total_hits: int) -> List[float]:
    """
    Applies the weighted running average for Phase 4.
    """
    v_old = np.array(old_vector)
    v_new = np.array(new_sample)
    
    # Math: (V_old * hits + V_new) / (hits + 1)
    nudged = (v_old * total_hits + v_new) / (total_hits + 1)
    
    # Normalize to maintain unit length
    norm = np.linalg.norm(nudged)
    if norm < 1e-9:
        new_norm = np.linalg.norm(v_new)
        return (v_new / new_norm).tolist() if new_norm > 1e-9 else v_new.tolist()
    return (nudged / norm).tolist()