from rag import load_store, retrieve_top_k

if __name__ == "__main__":
    embeddings, docs, meta = load_store()

    query = "database connection timeout"

    results = retrieve_top_k(
        query=query,
        embeddings=embeddings,
        docs=docs,
        meta=meta,
        k=3
    )

    print("\nTop results:\n")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"{r['meta']['source']} ({r['meta']['type']})")
        print(r["text"][:300])
        print("-" * 60)