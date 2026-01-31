## AIOps Incident RAG System

As a DevOps engineer, incidents and outages are inevitable. Performing Root Cause Analysis (RCA) during production issues is often time-consuming, and delays can significantly increase the impact and severity of outages.

This project demonstrates how AI-powered semantic search (RAG) can be used to speed up RCA and incident resolution by leveraging historical incidents and operational runbooks.

## Solution Overview

The system uses Retrieval-Augmented Generation (RAG) principles to retrieve the most relevant past incidents and documentation based on a user query.

Knowledge Sources

Incidents: Historical incident reports

Runbooks: Operational and troubleshooting documentation

These are embedded once and stored as vectors, allowing fast semantic retrieval during an incident.

How It Works

Incident reports and runbooks are stored under the knowledge/ directory.

The embed_knowledge.py script:

Reads all files

Chunks them into manageable pieces

Converts text into vector embeddings

Stores embeddings and metadata locally

When a user provides a query:

The query is embedded

Compared against stored embeddings using cosine similarity

The Top-K most relevant chunks are returned

This enables engineers to quickly find similar incidents and relevant runbook steps.

## Project Structure

aiops-incident-rag/
│
├── knowledge/
│ ├── incidents/ # Historical incident reports
│ └── runbooks/ # Operational runbooks
│
├── src/
│ ├── embed_knowledge.py # Ingest, chunk, and embed knowledge (run once)
│ ├── rag.py # Vector loading and top-K semantic retrieval
│ └── main.py # Query entry point
│
├── embeddings.npy # Stored embedding vectors
├── docs.npy # Chunked document text
├── meta.npy # Metadata for each chunk
│
├── README.md
└── requirements.txt
How to Run

---

Step 1: Build the Knowledge Base

Run this once (or whenever knowledge files change):

python embed_knowledge.py
Step 2: Query the System
python main.py

Modify the query and K value inside main.py to control retrieval behavior.

## Use Case

Faster RCA during production incidents

Reduced Mean Time to Resolution (MTTR)

Easy access to relevant runbooks and historical incidents

Foundation for a full RAG-based incident assistant

## Current Constraints and Future deveopment

There should be more data which should be present in the embedding DB along with the steps which were taken to solve those incidents

The prompt has to be made better to be more precise and better embeddings model
