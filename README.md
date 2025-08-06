Markdown

# LLM Document Processing System

## 🌟 Overview

This project implements an end-to-end system for processing natural language queries against large, unstructured documents using Large Language Models (LLMs). The system is designed to intelligently parse queries, semantically retrieve relevant information from a knowledge base of documents (e.g., policy documents, contracts), and generate a structured, justifiable decision in JSON format.

The core of the system is a **Retrieval-Augmented Generation (RAG)** pipeline, which leverages vector embeddings and a local vector store to provide the LLM with the most relevant context, moving beyond simple keyword matching.

## 🚀 Features

* **Intelligent Query Parsing**: Processes complex, plain-English queries to identify key details.
* **Semantic Search**: Retrieves relevant document clauses based on meaning, not just keywords.
* **Structured Output**: Returns a consistent JSON response containing a decision, amount, and justification.
* **Traceable Justification**: Explains decisions by referencing the exact clauses from the source documents.
* **Modular Architecture**: Designed for easy swapping of components, such as the LLM or vector store.

## 💻 Technical Architecture

The system's workflow follows a modular RAG pipeline:

1.  **Document Ingestion**: PDFs are parsed into raw text and then split into smaller, semantically meaningful chunks.
2.  **Embedding & Indexing**: Each text chunk is converted into a high-dimensional vector embedding using a powerful embedding model (`BAAI/bge-large-en-v1.5`). These embeddings are stored in a local **FAISS** vector index for efficient searching.
3.  **Retrieval**: A user query is also embedded, and the FAISS index is searched to retrieve the top `k` most relevant text chunks.
4.  **LLM Generation**: The retrieved chunks and the original query are passed to a **Qwen** LLM. The LLM acts as a reasoning engine, using the provided context to generate a final, structured JSON response.

## 🛠️ Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.10 or higher
* Git
* Conda or a Python virtual environment manager

### 1. Clone the Repository

Clone the project from GitHub and navigate into the directory.

```bash
git clone [https://github.com/Nikhil18207/Bajaj-Hackathon.git](https://github.com/Nikhil18207/Bajaj-Hackathon.git)
cd Bajaj-Hackathon
2. Set Up the Environment
Create and activate a Python virtual environment.

Bash

# For Conda
conda create -n llm-processing python=3.10
conda activate llm-processing

# For venv
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
3. Install Dependencies
Install all required packages from the requirements.txt file.

Bash

pip install -r requirements.txt
4. Run the Data Processing Pipeline
First, place your raw documents (PDFs) into the data/raw_pdfs/ folder. Then, run the following scripts in order to parse the documents, generate embeddings, and build the FAISS index.

Bash

# 1. Parse PDFs into text files
python parsing/pdf_parser.py

# 2. Build and save the FAISS index
python retrieval/faiss_store.py
5. Run the Core RAG System
The following command will run the full RAG pipeline, from retrieving chunks to generating a final JSON response.

Bash

python llm.py
