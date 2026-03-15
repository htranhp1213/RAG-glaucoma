# OphthoRAG: Retrieval-Augmented Generation for Ophthalmology Research

OphthoRAG is a **domain-specific Retrieval-Augmented Generation (RAG) system** designed to support research in **artificial intelligence for ophthalmology**. The system integrates a curated corpus of scientific papers, datasets, and clinical AI studies related to **eye diseases and ophthalmic medical imaging**, including both **left-eye and right-eye imaging data**.

By combining **vector search, document retrieval, and large language models**, OphthoRAG enables researchers to efficiently explore **multimodal AI methods, datasets, and clinical applications in ophthalmology**, while ensuring that answers are **grounded in verifiable scientific literature**.

The system supports both:

- **CLI / backend usage** for experimentation and evaluation  
- **Interactive UI** for question answering over ophthalmology literature  

---

# System Architecture

OphthoRAG follows a **modular RAG architecture** composed of four major components:

```
User
 ↓
Frontend (React QA Interface)
 ↓
Backend API (FastAPI)
 ↓
RAG Engine
 ↓
Vector Database + Document Corpus
```

---

# 1. Frontend (React UI)

The frontend provides a **web-based question answering interface** for interacting with the system.

### Responsibilities

- Accept ophthalmology questions from users
- Send queries to the backend API
- Display generated answers
- Show retrieved source passages and citations

### Technologies

- React
- Vite
- REST API communication with FastAPI

### Example Workflow

1. User enters a glaucoma-related question  
2. Frontend sends query to `/ask` API  
3. Backend processes the request through the RAG pipeline  
4. Answer and supporting sources are returned and displayed  

---

# 2. Backend (FastAPI)

The backend acts as the **orchestration layer** connecting the UI and the RAG engine.

### Responsibilities

- Receive user queries
- Call the RAG retrieval pipeline
- Format the response
- Return answers with supporting evidence

### Technology

- FastAPI

### Example API Request

```
POST /ask
{
  "query": "What are common treatments for glaucoma?"
}
```

### Backend Flow

1. Receive query from frontend  
2. Send query to RAG engine  
3. Retrieve relevant passages  
4. Send context to LLM  
5. Return grounded answer + sources  

---

# 3. RAG Engine

The RAG engine performs the **core retrieval and generation process**.

### Query Processing

The user query is converted into a vector representation using an embedding model.

Example embedding models:

- Sentence Transformers
- HuggingFace embeddings

---

### Document Retrieval

Relevant documents are retrieved from the vector database.

Steps:

1. Convert query into embedding  
2. Perform similarity search  
3. Retrieve **Top-K relevant document chunks**

Each chunk includes metadata such as:

- Source document
- Page number
- Section text

---

### Context Selection

Retrieved chunks are ranked and filtered.

Typical settings:

```
Top-K retrieval: 5–10 chunks
Chunk size: ~500 tokens
Overlap: ~100 tokens
```

---

### Answer Generation

Selected context is passed to a **Large Language Model (LLM)** with the user query.

Example prompt template:

```
Answer the question using ONLY the provided context.
If the answer cannot be found in the context, say so.

Context:
{retrieved_passages}

Question:
{user_query}
```

---

# 4. Vector Database

OphthoRAG uses a vector database to store document embeddings.

Current setup:

- **ChromaDB vector store**

Stored metadata includes:

- Document name
- Page number
- Section text
- Additional modality metadata

Example stored record:

```
{
  text: "... glaucoma treatment includes lowering intraocular pressure ...",
  metadata: {
      source: "Primary Open Angle Glaucoma PPP",
      page: 63
  }
}
```

---

# Document Ingestion Pipeline

The ingestion pipeline prepares ophthalmology documents for retrieval.

### Steps

1. Load ophthalmology PDFs  
2. Split documents into chunks  
3. Generate embeddings  
4. Store embeddings in vector database  

### Scripts

```
ingest_to_chroma.py
add_multimodal_records.py
```

Example output:

```
[OK] Ingested 2029 records
[OK] Total records now: 2517
```

---

# How to Run the System

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ophthorag.git
cd ophthorag
```

---

# Running Without UI (Backend / CLI)

This mode allows testing retrieval directly.

## Step 1 — Install Dependencies

Create environment:

```bash
conda create -n ophthorag python=3.9
conda activate ophthorag
```

Install packages:

```bash
pip install -r requirements.txt
```

---

## Step 2 — Ingest Documents

```bash
python ingest_to_chroma.py
python add_multimodal_records.py
```

This creates the vector database in:

```
vector_db/
```

---

## Step 3 — Test Retrieval

```bash
python test_rag_retrieve.py
```

Example output:

```
Q: What is the primary goal of glaucoma treatment?

Top result:
Primary Open-Angle Glaucoma PPP.pdf | page=24
```

---

# Running With UI

## Step 1 — Start Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

## Step 2 — Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## Step 3 — Ask Questions

Example queries:

```
What is the primary goal of glaucoma treatment?

How is intraocular pressure managed in glaucoma?

What datasets exist for retinal disease classification?
```

The system returns:

- Generated answer
- Supporting passages
- Source citations

---

# Evaluation

The system supports evaluation using **RAGAS metrics**.

Metrics include:

- Faithfulness
- Context Precision

Run evaluation:

```bash
python evaluate_rag.py
```

Output files:

```
eval_results_dense.csv
eval_summary_dense.json
```

---

# Repository Structure

```
OphthoRAG
│
├── backend
│   └── FastAPI server
│
├── frontend
│   └── React QA interface
│
├── vector_db
│   └── Chroma vector store
│
├── ingest_to_chroma.py
├── add_multimodal_records.py
├── test_rag_retrieve.py
├── compare_retrievers.py
│
└── README.md
```

---

# Limitations

- The system depends on the quality and coverage of the document corpus  
- LLM responses may hallucinate if insufficient context is retrieved  
- Current retrieval focuses mainly on **text-based content**

### Future Work

- Integrate **multimodal ophthalmology datasets**
- Improve retrieval ranking
- Add **hybrid dense + sparse retrieval**

---

# Citation

If you use OphthoRAG in research, please cite:

```
@software{ophthorag2026,
  title = {OphthoRAG: A Retrieval-Augmented Generation System for Ophthalmology Research},
  author = {Tran, Linh Hy},
  year = {2026},
  url = {https://github.com/yourusername/ophthorag}
}
```
