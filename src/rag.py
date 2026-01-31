import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# Paths (always resolve from project root)
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
DOCS_PATH = os.path.join(BASE_DIR, "docs.npy")
META_PATH = os.path.join(BASE_DIR, "meta.npy")

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# -------------------------------------------------------------------
# Load stored knowledge base
# -------------------------------------------------------------------
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        "embeddings.npy not found. Run embed_knowledge.py first."
    )

embeddings = np.load(EMBEDDINGS_PATH)
docs = np.load(DOCS_PATH, allow_pickle=True)
meta = np.load(META_PATH, allow_pickle=True)

# -------------------------------------------------------------------
# Retrieval function
# -------------------------------------------------------------------
def retrieve(query: str, k: int = 3):
    """
    Retrieve top-K most relevant chunks for a query.

    Returns:
        List[dict] with keys:
        - text
        - source
        - type
        - chunk
        - score
    """
    # Encode query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Compute cosine similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]

    # Top-K indices (highest score first)
    top_k_indices = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append({
            "text": docs[idx],
            "score": float(scores[idx]),
            "source": meta[idx]["source"],
            "type": meta[idx]["type"],
            "chunk": meta[idx]["chunk"],
        })

    return results