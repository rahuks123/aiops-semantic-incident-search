🔧 AIOps Incident RAG System

As a DevOps engineer, incidents and outages are inevitable. Performing Root Cause Analysis (RCA) during production issues is often time-consuming, and delays can significantly increase the impact and severity of outages.

This project demonstrates how AI-powered semantic search using Retrieval-Augmented Generation (RAG) can accelerate RCA and reduce resolution time by leveraging historical incidents and operational runbooks.

🧠 Solution Overview

The system applies RAG principles to retrieve the most relevant past incidents and documentation based on a user’s query.

Knowledge Sources

Incidents
Historical incident reports from previous outages

Runbooks
Operational and troubleshooting documentation

These documents are embedded once and stored as vectors, enabling fast semantic retrieval during an incident.

⚙️ How It Works

1. Knowledge Ingestion

All incident reports and runbooks are stored under the knowledge/ directory.

The embed_knowledge.py script:

Reads all knowledge files

Chunks documents into manageable pieces

Converts text into vector embeddings

Stores embeddings and metadata locally

This step is executed once (or whenever knowledge changes).

2. Query & Retrieval

When a user provides a query:

The query is converted into an embedding

It is compared against stored embeddings using cosine similarity

The Top-K most relevant chunks are retrieved

This allows engineers to quickly surface:

Similar historical incidents

Relevant runbook steps

Operational context for faster RCA

📁 Project Structure
aiops-incident-rag/
│
├── knowledge/
│ ├── incidents/ # Historical incident reports
│ └── runbooks/ # Operational runbooks
│
├── src/
│ ├── embed_knowledge.py # Ingest, chunk, and embed knowledge (run once)
│ ├── rag.py # Vector loading and Top-K semantic retrieval
│ └── main.py # Query entry point
│
├── embeddings.npy # Stored embedding vectors
├── docs.npy # Chunked document text
├── meta.npy # Metadata for each chunk
│
├── README.md
└── requirements.txt

ℹ️ The folder tree is wrapped in a fenced code block to ensure proper alignment on GitHub.

▶️ How to Run
Step 1: Build the Knowledge Base

Run this once (or whenever knowledge files change):

python embed_knowledge.py

Step 2: Query the System
python main.py

Modify the query and Top-K value inside main.py to control retrieval behavior.

🎯 Use Cases

Faster RCA during production incidents

Reduced Mean Time to Resolution (MTTR)

Easy access to relevant runbooks and historical incidents

Foundation for a full AI-powered incident assistant

🚧 Current Constraints & Future Enhancements

Increase the volume and diversity of incident and runbook data

Store detailed remediation steps taken during past incidents

Improve prompt design for more precise and actionable responses

Experiment with stronger embedding models and hybrid retrieval

Integrate an LLM for grounded RCA + general troubleshooting guidance
