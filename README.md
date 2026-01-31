## AIOps Incident RAG System

As a DevOps engineer, incidents and outages are inevitable. Performing Root Cause Analysis (RCA) during production issues is often time-consuming, and delays can significantly increase the impact and severity of outages.

This project demonstrates how AI-powered semantic search (RAG) can be used to speed up RCA and incident resolution by leveraging historical incidents and operational runbooks.

## Solution Overview

The system uses Retrieval-Augmented Generation (RAG) principles to retrieve the most relevant historical incidents and documentation based on a user query.

Knowledge Sources

Incidents
Historical incident reports

Runbooks
Operational and troubleshooting documentation

These documents are embedded once and stored as vectors, enabling fast semantic retrieval during an incident.

⚙️ How It Works

Incident reports and runbooks are stored under the knowledge/ directory and serve as the system’s knowledge base. The embed_knowledge.py script processes these files by reading their contents, splitting them into manageable chunks, converting each chunk into a vector embedding, and storing the resulting embeddings along with their associated metadata locally. This ingestion step is performed once and only needs to be rerun when the knowledge files are updated.

When a user submits a query, the system converts the query into an embedding and compares it against the stored embeddings using cosine similarity. Based on this comparison, the Top-K most relevant chunks are retrieved and returned. This allows engineers to quickly surface similar historical incidents and relevant runbook steps, enabling faster and more informed root cause analysis during incidents.

## Project Structure

aiops-incident-rag/

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

<br>
▶️ How to Run
<br>
Step 1: Build the Knowledge Base

Run this once (or whenever knowledge files change):

python embed_knowledge.py

Step 2: Query the System
python main.py

Modify the query and Top-K value inside main.py to control retrieval behavior.

## Use Case

Faster RCA during production incidents

Reduced Mean Time to Resolution (MTTR)

Easy access to relevant runbooks and historical incidents

Foundation for a full RAG-based incident assistant

## Current Constraints and Future deveopment

There should be more data which should be present in the embedding DB along with the steps which were taken to solve those incidents

The prompt has to be made better to be more precise and better embeddings model
