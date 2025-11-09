import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(emb, known_embs):
    """Najde nejpodobnějšího známého mluvčího podle embeddingu"""
    best_name = "unknown"
    best_score = 0.0
    for name, ref in known_embs.items():
        score = cosine_similarity(emb, ref)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score > 0.7 else "speaker_?"
