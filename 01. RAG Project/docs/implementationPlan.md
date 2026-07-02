# Ask My PDF — Implementation Plan

This document breaks the project into **10 small phases**. Each phase builds on the previous one, and by the end you will have a fully working RAG application.

**How to use this plan:**
- Complete one phase at a time.
- Do not skip ahead — each phase depends on the previous one.
- Run the validation checklist at the end of each phase before moving on.
- If something breaks, debug it in the current phase before continuing.

```
Phase 1        Phase 2        Phase 3        Phase 4         Phase 5
Project   →    PDF       →    Text      →    Chunking   →    Embedding
Setup          Loading        Extraction                     Generation

Phase 6        Phase 7        Phase 8        Phase 9         Phase 10
Vector    →    User      →    Question  →    Retrieval  →    Answer
Database       Question       Embedding                     Generation
```

---

## Phase 1: Project Setup

### Goal

Set up the project folder structure, virtual environment, and install all required dependencies so you have a clean foundation to build on.

### What to Implement

- Create the project directory structure:
  ```
  ask-my-pdf/
  ├── app.py              # Main Streamlit application (empty for now)
  ├── requirements.txt    # All project dependencies
  ├── .env                # API keys (Groq)
  ├── .gitignore          # Ignore .env, __pycache__, venv, etc.
  └── docs/               # Documentation (already exists)
  ```
- Create a Python virtual environment.
- Create `requirements.txt` with all dependencies:
  - `streamlit` — UI framework
  - `pypdf2` — PDF text extraction
  - `langchain` — Orchestration and text splitting
  - `langchain-community` — Community integrations
  - `langchain-huggingface` — HuggingFace embedding integration
  - `chromadb` — Vector database
  - `langchain-groq` — Groq LLM integration
  - `python-dotenv` — Load environment variables from `.env`
- Install all dependencies.
- Create a `.env` file with a placeholder for `GROQ_API_KEY`.
- Create a minimal `app.py` that runs Streamlit and displays "Ask My PDF" as the page title.

### Expected Output

- Running `streamlit run app.py` opens a browser page displaying "Ask My PDF".
- All dependencies install without errors.

### Validation Checklist

- [ ] Virtual environment is created and activated.
- [ ] `pip install -r requirements.txt` runs successfully with no errors.
- [ ] `.env` file exists with `GROQ_API_KEY=your_key_here`.
- [ ] `.gitignore` excludes `.env`, `__pycache__/`, and `venv/`.
- [ ] `streamlit run app.py` launches and shows the page title in the browser.

---

## Phase 2: PDF Loading

### Goal

Allow the user to upload a single PDF file through the Streamlit UI. At this stage, you are only loading the file — not processing it yet.

### What to Implement

- Add a file uploader widget in `app.py` using `st.file_uploader()`.
  - Accept only `.pdf` files.
  - Limit to a single file upload.
- When a PDF is uploaded, display:
  - The file name.
  - The file size (in KB).
  - A success message: "PDF uploaded successfully!"
- When no PDF is uploaded, show a message: "Please upload a PDF to get started."

### Expected Output

- The UI shows a file upload area.
- Uploading a PDF displays its name and size.
- Uploading a non-PDF file is rejected by the widget.

### Validation Checklist

- [ ] The file uploader appears on the page.
- [ ] Only `.pdf` files are accepted.
- [ ] Uploading a PDF shows the file name and size.
- [ ] The success message appears after upload.
- [ ] The prompt message appears when no file is uploaded.

---

## Phase 3: Text Extraction

### Goal

Extract raw text from the uploaded PDF so it can be processed by the rest of the pipeline.

### What to Implement

- Use `PyPDF2` (PdfReader) to read the uploaded PDF file.
- Extract text from every page of the PDF.
- Combine all page text into a single string.
- Display the extracted text in an expandable section (`st.expander`) for debugging.
- Show the total character count of the extracted text.

### Expected Output

- After uploading a PDF, the full extracted text is visible in an expandable section.
- The character count is displayed (e.g., "Extracted 12,450 characters").

### Validation Checklist

- [ ] Text is extracted from all pages of the PDF.
- [ ] The extracted text is displayed in an expandable section.
- [ ] The character count is shown.
- [ ] Uploading a different PDF replaces the previously extracted text.
- [ ] A text-based PDF (not scanned) extracts readable content.

---

## Phase 4: Chunking

### Goal

Split the extracted text into smaller, overlapping chunks. This is necessary because embedding models and LLMs have input size limits, and smaller chunks improve retrieval precision.

### What to Implement

- Use LangChain's `RecursiveCharacterTextSplitter` to split the text.
- Configure the splitter with:
  - `chunk_size=500` — Each chunk is approximately 500 characters.
  - `chunk_overlap=50` — Adjacent chunks share 50 characters to avoid cutting sentences.
- After splitting, display:
  - Total number of chunks created.
  - The first 3 chunks in an expandable section (for debugging).

### Expected Output

- After text extraction, the text is split into chunks.
- The UI shows how many chunks were created (e.g., "Created 25 chunks").
- The first 3 chunks are visible in an expandable section.

### Validation Checklist

- [ ] The text is split into multiple chunks.
- [ ] Each chunk is approximately 500 characters long.
- [ ] Chunks overlap by approximately 50 characters (check by comparing the end of chunk N with the start of chunk N+1).
- [ ] The total number of chunks is displayed.
- [ ] The first 3 chunks are displayed for inspection.

---

## Phase 5: Embedding Generation (BGE)

### Goal

Convert each text chunk into a vector embedding using the BGE model. These embeddings represent the semantic meaning of each chunk as a list of numbers.

### What to Implement

- Use `langchain-huggingface`'s `HuggingFaceEmbeddings` to load the BGE model.
  - Model name: `BAAI/bge-small-en-v1.5`
  - This model runs locally — no API key needed.
- Generate embeddings for all chunks.
- Display:
  - The embedding dimension (e.g., "Embedding dimension: 384").
  - The embedding of the first chunk (truncated to the first 5 values) for debugging.

### Learning & Debugging

- Save all generated embeddings to a local debug file (`debug/embeddings.json`).
- The file should contain each chunk's text alongside its embedding vector.
- Example structure:
  ```json
  [
    {
      "chunk_index": 0,
      "chunk_text": "First 100 characters of the chunk...",
      "embedding": [0.0231, -0.0412, 0.0187, ...],
      "embedding_dimension": 384
    }
  ]
  ```
- **Why this file exists:** This debug file is only for learning and verification. It lets you inspect the raw embedding vectors before they go into the vector database. You can open it, compare embeddings across chunks, and build intuition for what embeddings look like. This file is **not used** by the application at runtime — it is purely a learning aid.
- **Tip:** Try comparing the embeddings of two similar chunks vs. two very different chunks. Similar chunks will have vectors that are closer together.

### Expected Output

- Each chunk is converted to a vector.
- The UI shows the embedding dimension and a preview of the first embedding.
- The first time you run this, the BGE model will be downloaded automatically (~130 MB).
- A `debug/embeddings.json` file is created with all chunk-embedding pairs.

### Validation Checklist

- [ ] The BGE model loads successfully (first run may take a minute to download).
- [ ] Each chunk produces an embedding of the same dimension (384 for bge-small-en-v1.5).
- [ ] The number of embeddings matches the number of chunks.
- [ ] The embedding dimension is displayed.
- [ ] A preview of the first embedding is shown.
- [ ] `debug/embeddings.json` is created and contains all chunks with their embeddings.
- [ ] Opening the debug file shows readable chunk text alongside embedding vectors.

---

## Phase 6: Vector Database (ChromaDB)

### Goal

Store the chunk embeddings in ChromaDB (an in-memory vector database) so they can be searched later by similarity.

### What to Implement

- Use ChromaDB as the vector store via LangChain's `Chroma` integration.
- Store each chunk's text and its embedding in the ChromaDB collection.
- Use the same BGE embedding function from Phase 5 when creating the vector store.
- Display:
  - A success message: "Stored {N} chunks in the vector database."
  - The total number of documents in the collection.
- After storing, display **one sample stored document** in an expandable debug section:
  - The document text (chunk content).
  - Its metadata (e.g., chunk index, source).
  - Its document ID (the unique ID assigned by ChromaDB).
- **Why show a sample document?** This helps learners understand what is actually stored inside a vector database. It's not just embeddings — it's the text, metadata, and an ID. Seeing a real stored document demystifies what "storing in a vector DB" means.

### Expected Output

- All chunks and their embeddings are stored in ChromaDB.
- The UI confirms how many chunks were stored.
- A sample document is displayed with its text, metadata, and document ID.

### Validation Checklist

- [ ] ChromaDB collection is created without errors.
- [ ] The number of stored documents matches the number of chunks.
- [ ] The success message displays the correct count.
- [ ] One sample document is displayed with its text, metadata, and document ID.
- [ ] Uploading a new PDF re-creates the vector store (no stale data from a previous PDF).

---

## Phase 7: User Question

### Goal

Add a text input field where the user can type a question about the uploaded PDF.

### What to Implement

- Add a text input widget (`st.text_input`) below the PDF upload section.
- The text input should:
  - Be disabled until a PDF is uploaded and indexed.
  - Show a placeholder: "Ask a question about your PDF..."
- When the user types a question and presses Enter, display the question back to them (for now, no answer yet).

### Expected Output

- A text input field appears after the PDF is indexed.
- Typing a question and pressing Enter shows the question on screen.

### Validation Checklist

- [ ] The text input is visible after PDF upload and indexing.
- [ ] The text input is disabled (or hidden) when no PDF is uploaded.
- [ ] The placeholder text is displayed.
- [ ] The user's question is displayed after submission.

---

## Phase 8: Question Embedding

### Goal

Convert the user's question into a vector embedding using the **same** BGE model used for the chunks. This is essential — the question and the chunks must be in the same vector space for similarity search to work.

### What to Implement

- When the user submits a question, embed it using the same BGE model from Phase 5.
- Display (in an expandable debug section):
  - The question text.
  - The embedding dimension (should match the chunk embeddings — 384).
  - The first 5 values of the question embedding.

### Expected Output

- The user's question is embedded into a vector.
- Debug info confirms the embedding dimension matches the chunk embeddings.

### Validation Checklist

- [ ] The question embedding uses the same BGE model as the chunk embeddings.
- [ ] The embedding dimension matches (384).
- [ ] The debug section shows the question and its embedding preview.
- [ ] Different questions produce different embeddings.

---

## Phase 9: Retrieval

### Goal

Use the question embedding to search the vector database and retrieve the most relevant chunks from the PDF.

### What to Implement

- Query ChromaDB with the question embedding using a similarity search.
- Retrieve the top-K most relevant chunks (start with K=3).
- Display the retrieved chunks in an expandable debug section. For each retrieved chunk, show:
  - **Chunk text** — The actual text content of the chunk.
  - **Similarity score** — The distance/score between the question embedding and the chunk embedding (lower distance = more similar).
  - **Chunk ID** — The document ID assigned by ChromaDB.
  - **Why this chunk was retrieved** — A short explanation (e.g., "This chunk scored 0.23 similarity because it contains keywords and concepts related to your question about X"). This explanation is generated by you (the developer) as a debug label — not by the LLM.

- **Why show all this?** This debug view exists purely for learning. It lets you see exactly what the retrieval stage is doing — which chunks it chose, how confident it is (via the score), and whether it made a good selection. Understanding retrieval quality is key to understanding why RAG answers are good or bad.

### Expected Output

- Asking a question retrieves the 3 most relevant chunks from the PDF.
- Each retrieved chunk is displayed with its text, similarity score, and chunk ID.
- A brief explanation of why each chunk was retrieved is shown.

### Validation Checklist

- [ ] The similarity search returns exactly K chunks (3).
- [ ] The retrieved chunks are relevant to the question (read them to verify).
- [ ] Similarity scores are displayed alongside each chunk.
- [ ] Chunk IDs are displayed for each retrieved chunk.
- [ ] Different questions retrieve different chunks.
- [ ] If you ask a question completely unrelated to the PDF, the chunks still return (but with lower similarity scores).

---

## Phase 10: Answer Generation using Groq

### Goal

Send the user's question along with the retrieved chunks to the Groq LLM API to generate a final answer grounded in the PDF content.

### What to Implement

- Load the `GROQ_API_KEY` from the `.env` file using `python-dotenv`.
- Create a Groq LLM client using LangChain's `ChatGroq`.
  - Model: `llama-3.1-8b-instant` (or another model available on Groq's free tier).
- Build a prompt that includes:
  - A system instruction: "Answer the question based only on the provided context. If the context does not contain the answer, say 'I don't have enough information to answer this question.'"
  - The retrieved chunks as context.
  - The user's question.
- Send the prompt to Groq and display the generated answer.
- Display the answer in the main UI area.

### Expected Output

- The user asks a question, and the system returns a natural language answer.
- The answer is clearly based on the PDF content, not the LLM's general knowledge.
- If the question is outside the PDF's scope, the system says it doesn't have enough information.

### Validation Checklist

- [ ] The Groq API key is loaded from `.env` successfully.
- [ ] The prompt includes the system instruction, retrieved chunks, and user question.
- [ ] The answer is displayed in the UI.
- [ ] The answer is grounded in the PDF content (compare with the retrieved chunks).
- [ ] Asking a question unrelated to the PDF produces an "I don't have enough information" response (or similar).
- [ ] The complete pipeline works end-to-end: Upload PDF → Ask Question → Get Answer.

---

## Summary: The Complete RAG Pipeline

After completing all 10 phases, your application will implement the full RAG pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                            │
│                                                                     │
│   Upload PDF → Extract Text → Chunk Text → Embed Chunks → Store    │
│   (Phase 2)    (Phase 3)      (Phase 4)    (Phase 5)     (Phase 6) │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
                            Vector Database
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PIPELINE                            │
│                                                                     │
│   User Question → Embed Question → Retrieve Chunks → Generate      │
│   (Phase 7)       (Phase 8)        (Phase 9)         Answer        │
│                                                       (Phase 10)    │
└─────────────────────────────────────────────────────────────────────┘
```

> **You now understand every stage of the RAG pipeline. You didn't just build it — you can explain it.**

---

## Learning Deliverables

After completing **every phase**, create a learning file:

```
learning/phase-X.md
```

For example: `learning/phase-1.md`, `learning/phase-2.md`, ..., `learning/phase-10.md`.

Each learning file should contain the following sections:

| Section | What to Write |
|---------|---------------|
| **What was built?** | Describe what you implemented in this phase in 2-3 sentences. |
| **What problem does this solve?** | Explain the problem this phase addresses in the RAG pipeline. |
| **Why is this phase required?** | Explain what would break or be missing if you skipped this phase. |
| **Important concepts learned** | List the key technical concepts you learned (e.g., embeddings, vector similarity, chunking overlap). |
| **Interview questions (3-5)** | Write 3-5 questions an interviewer might ask about this topic. Include your answers. |
| **Common mistakes** | List mistakes a beginner might make in this phase and how to avoid them. |
| **Key takeaway (1-2 lines)** | Summarize the most important thing you learned in this phase. |

### Example: `learning/phase-4.md`

```markdown
# Phase 4: Chunking

## What was built?
Split the extracted PDF text into smaller overlapping chunks using
RecursiveCharacterTextSplitter with chunk_size=500 and chunk_overlap=50.

## What problem does this solve?
LLMs and embedding models have input size limits. Chunking also
improves retrieval precision — smaller chunks mean more targeted results.

## Why is this phase required?
Without chunking, the entire document would be one giant block of text.
Embedding the whole document as one vector would lose the ability to
find specific paragraphs that answer a question.

## Important concepts learned
- Chunking strategies (fixed-size, recursive, semantic)
- Chunk overlap and why it prevents information loss at boundaries
- Trade-off between chunk size and retrieval precision

## Interview questions
1. What is chunking and why is it needed in RAG?
2. What happens if your chunks are too large? Too small?
3. What is chunk overlap and why is it important?
4. What chunking strategies exist beyond fixed-size?

## Common mistakes
- Setting chunk_overlap to 0 — cuts sentences at boundaries.
- Making chunks too large — reduces retrieval precision.
- Making chunks too small — loses context needed for understanding.

## Key takeaway
Chunking is where you decide the granularity of your knowledge base.
The right chunk size directly impacts retrieval quality and answer accuracy.
```

### Purpose

These files serve as **interview notes built alongside the project**. By the time you finish all 10 phases, you will have a complete set of notes covering every stage of the RAG pipeline — ready for revision, interviews, or teaching someone else.
