# Lab 3: Build a Local RAG Prototype with ChromaDB

**Goal:** Create a local vector store from a text file and perform semantic search.
**Audience:** DevOps engineers with basic Python familiarity.
**Time:** 15-20 Minutes

---

## 1. Lab Objectives

By the end of this lab, you will understand the mechanics of **Retrieval-Augmented Generation (RAG)** by building it from scratch on your laptop. You will:
1.  **Ingest** raw text data (simulation of a company handbook).
2.  **Embed** that text into vectors using `sentence-transformers`.
3.  **Store** those vectors in a local instance of **ChromaDB**.
4.  **Query** the database to find answers based on meaning, not just keywords.

---

## 2. Environment Setup

We will use a purely local stack. No OpenAI keys or cloud APIS are required.

### Prerequisites
Make sure you have Python installed. Then, create a virtual environment and install the required libraries:

```bash
# Create and activate virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install chromadb sentence-transformers
```

---

## 3. The Implementation

We will build this in a single Python script called `local_rag.py`.

### Step 1: Prepare Your Data
First, create a dummy text file named `handbook.txt` in the same folder. This represents your "Long-Term Memory."

**Content for `handbook.txt`:**
```text
The payment service runs on port 8080.
To restart the payment service, use the command: kubectl rollout restart deployment payments.
The database backup runs everyday at 2 AM UTC.
If the API returns a 500 error, check the Redis connection string in the secret manifest.
The admin dashboard is accessible at admin.internal.company.com.
```

### Step 2: The RAG Script (`local_rag.py`)

Copy the following code into `local_rag.py`. Read the comments carefully—they explain the "why" behind every line.

```python
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
DATA_FILE = "handbook.txt"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_knowledge"

def load_documents(file_path):
    """
    Reads the text file and splits it into lines (chunks).
    In a real system, you would use a sophisticated chunking strategy.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []
    
    with open(file_path, "r") as f:
        text = f.read()
    
    # Simple chunking by newlines for this lab
    chunks = [line.strip() for line in text.split('\n') if line.strip()]
    return chunks

def main():
    # 1. Initialize ChromaDB (Persistent Client)
    # This creates a folder named 'chroma_db' to store the vectors on disk.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 2. Select an Embedding Model
    # We use a free, local model from HuggingFace.
    # It converts text into a list of numbers (vector).
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Create or Get Collection
    # A 'Collection' is like a table in SQL.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    # 4. Ingest Data (Only if collection is empty or you want to refresh)
    # real-world apps would check for duplicates here.
    if collection.count() == 0:
        print("--- Indexing Data into Vector Store ---")
        chunks = load_documents(DATA_FILE)
        
        ids = [f"id_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "handbook.txt"} for _ in chunks]
        
        # This is where the magic happens:
        # Chroma takes the 'documents' (text), runs them through the 'embedding_function',
        # and stores the resulting vectors.
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully stored {len(chunks)} chunks.")
    else:
        print("--- Data already indexed. Skipping ingestion. ---")

    # 5. The Query Loop
    print("\n--- RAG System Ready (Type 'exit' to quit) ---")
    while True:
        query_text = input("\nMy Question: ")
        if query_text.lower() == "exit":
            break
            
        # Perform Similarity Search
        # n_results=1 means "Give me the SINGLE most relevant chunk"
        results = collection.query(
            query_texts=[query_text],
            n_results=1
        )
        
        # Display Results
        if results['documents']:
            best_match = results['documents'][0][0]
            print(f"\nRAG Retrieved Context: \"{best_match}\"")
        else:
            print("\nNo relevant information found.")

if __name__ == "__main__":
    main()
```

---

## 4. Running the Lab

1.  Run the script:
    ```bash
    python local_rag.py
    ```
2.  The first time you run it, it will download the embedding model (approx. 80MB) and build the index.
3.  **Test Semantic Search:**
    *   Ask: *"How do I fix the database?"*
        *   *Expected Match:* The backup schedule or connection string, depending on phrasing.
    *   Ask: *"Payment system is down."*
        *   *Expected Match:* "To restart the payment service..." (Note: You didn't use the word 'restart' or 'deployment', but it understood the context of 'payment').
    *   Ask: *"Who is the admin?"*
        *   *Expected Match:* The admin dashboard URL.

---

## 5. Reflection & Operational Analysis

Congratulations! You just built a stateful AI component.
Now, put your **DevOps Hat** on and think about the implications of running this in production on Kubernetes.

### 1. The "State" Problem
You just created a folder called `chroma_db`. If you ran this as a microservice in a Kubernetes Pod:
*   **Without a PVC (Persistent Volume Claim):** That folder disappears when the pod restarts. Your agent gets amnesia.
*   **With a PVC:** You are now managing stateful workloads. Scaling becomes harder (ReadWriteOnce vs ReadWriteMany).

### 2. Ingestion Latency
In this lab, ingestion took milliseconds.
If `handbook.txt` was **10GB** of PDF files:
*   The embedding process requires heavy CPU/GPU.
*   The embedding service might need to be a separate `Job` or `Worker` to avoid blocking the API.

### 3. Data Freshness
If you edit `handbook.txt` right now and add a new line, the script **won't know** until you delete the `chroma_db` folder or write code to update the index.
In production, you need a **Real-time Sync** pipeline to detect changes in knowledge and push them to the Vector DB immediately.

### 4. Search Quality
We used a simple "chunk by line" strategy.
Real documents have headers, tables, and complex structures. "Chunking" is an entire engineering discipline itself. Bad chunking = Bad retrieval = Stupid Agent.

---
**End of Lab 3**

