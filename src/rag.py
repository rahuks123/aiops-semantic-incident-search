import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
DOCS_PATH = os.path.join(BASE_DIR, "docs.npy")
META_PATH = os.path.join(BASE_DIR, "meta.npy")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_store():
    embeddings = np.load(EMBEDDINGS_PATH)
    docs = np.load(DOCS_PATH, allow_pickle=True)
    meta = np.load(META_PATH, allow_pickle=True)
    return embeddings, docs, meta


def retrieve_top_k(query, embeddings, docs, meta, k=3):
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(scores)[-k:][::-1]

    return [
        {
            "score": float(scores[i]),
            "text": docs[i],
            "meta": meta[i],
        }
        for i in top_k_indices
    ]