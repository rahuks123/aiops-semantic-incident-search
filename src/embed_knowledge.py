import os
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_DIRS = {
    "incident": os.path.join(BASE_DIR, "knowledge", "incidents"),
    "runbook": os.path.join(BASE_DIR, "knowledge", "runbooks"),
}

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
DOCS_PATH = os.path.join(BASE_DIR, "docs.npy")
META_PATH = os.path.join(BASE_DIR, "meta.npy")

MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_documents():
    model = SentenceTransformer(MODEL_NAME)

    docs = []
    meta = []

    for doc_type, folder in KNOWLEDGE_DIRS.items():
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                meta.append({
                    "source": filename,
                    "type": doc_type,
                    "chunk": idx,
                })

    if not docs:
        raise ValueError("No documents found")

    embeddings = model.encode(
        docs,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(DOCS_PATH, np.array(docs, dtype=object))
    np.save(META_PATH, np.array(meta, dtype=object))

    print(f"Embedded {len(docs)} chunks and saved to disk")


if __name__ == "__main__":
    ingest_documents()