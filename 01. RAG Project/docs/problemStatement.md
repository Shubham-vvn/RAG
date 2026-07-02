# Ask My PDF — Problem Statement

## Overview

**Ask My PDF** is a beginner-friendly learning project that lets a user upload a single PDF and ask natural language questions about its content. The system answers **only** using information from the uploaded PDF, powered by Retrieval-Augmented Generation (RAG).

This project is designed for **learning**, not production. The scope is intentionally minimal so learners can focus on understanding the core RAG pipeline without getting lost in infrastructure or edge cases.

---

## Why RAG?

Large Language Models (LLMs) are powerful, but they have a key limitation: they can only respond based on the data they were trained on. They have no knowledge of your private documents, and when they don't know something, they may **hallucinate** — generate confident-sounding but incorrect answers.

**Retrieval-Augmented Generation (RAG)** solves this by adding a retrieval step before generation. Instead of relying solely on the LLM's internal knowledge, the system first searches the uploaded PDF for relevant content and then passes that content to the LLM as context. The LLM generates its answer **grounded in the retrieved text**, which significantly reduces hallucinations and keeps responses factually tied to the source document.

In short: **RAG = Retrieve first, then generate.** The LLM answers using your data, not its imagination.

---

## Project Goals

1. **Learn RAG fundamentals** — Understand how retrieval-augmented generation works end to end: document loading → chunking → embedding → vector storage → retrieval → answer generation.
2. **Build something working** — Produce a functional app where you upload a PDF, ask a question, and get an answer grounded in the document.
3. **Keep it simple** — Avoid over-engineering. One PDF at a time, one user, no auth, no database migrations, no deployment pipeline.
4. **Demystify LLM integration** — Get hands-on experience calling an LLM API and feeding it retrieved context.
5. **Understand every stage of the RAG pipeline** — Not just build a working application, but be able to explain what happens at each stage and why it matters.

---

## Target Users

- Developers learning about RAG for the first time.
- Students exploring how LLMs can be combined with external knowledge sources.
- Anyone who wants a minimal, runnable example to study and modify.

---

## Core Features (Scope)

The application follows the complete RAG pipeline:

```
Upload PDF → Extract Text → Chunk Text → Generate Embeddings → Store Embeddings → Retrieve Relevant Chunks → Generate Answer
```

| # | Pipeline Stage | Description |
|---|----------------|-------------|
| 1 | **Upload PDF** | User uploads a single PDF file through the UI. |
| 2 | **Extract Text** | Raw text is extracted from the PDF pages. |
| 3 | **Chunk Text** | The extracted text is split into smaller, overlapping chunks for better retrieval. |
| 4 | **Generate Embeddings** | Each chunk is converted into a vector embedding using an embedding model. |
| 5 | **Store Embeddings** | The embeddings are stored in an in-memory vector store for similarity search. |
| 6 | **Retrieve Relevant Chunks** | When the user asks a question, the most relevant chunks are retrieved based on semantic similarity. |
| 7 | **Generate Answer** | The retrieved chunks are passed as context to an LLM, which generates an answer grounded solely in the PDF content. |

---

## Success Criteria

- A user can upload a PDF and receive answers to questions about its content.
- Answers are clearly derived from the PDF, not from the LLM's general knowledge.
- The codebase is small enough that a beginner can read through it in one sitting.
- Each stage of the RAG pipeline (load → chunk → embed → store → retrieve → generate) is visible and understandable in the code.
- The learner should be able to explain every stage of the RAG pipeline after completing the project.

---

## Assumptions

- The user uploads **one PDF at a time**. There is no multi-document support.
- The PDF contains **extractable text** (not scanned images or handwritten notes).
- An **LLM API key** (e.g., OpenAI / Google Gemini) is available and configured.
- The project runs **locally** on the developer's machine.
- The vector store is **in-memory or file-based** — no external database required.

---

## Limitations

- **Single PDF only** — No support for uploading multiple documents or maintaining a knowledge base across sessions.
- **No authentication** — Anyone with access to the app can use it.
- **No conversation memory** — Each question is independent; there is no multi-turn chat context.
- **No production hardening** — No rate limiting, error recovery, logging infrastructure, or deployment setup.
- **Text-only PDFs** — PDFs with images, tables, or complex layouts may not be processed accurately.
- **Quality depends on chunking & retrieval** — Since this is a learning project, the chunking strategy and retrieval logic are kept simple and may not handle all edge cases well.
- **Incorrect answers from failed retrieval** — The assistant may provide incorrect answers if the retrieval stage fails to retrieve the correct chunks. This is a known trade-off in RAG systems.

---

## What This Project Is NOT

- ❌ A production-ready SaaS product.
- ❌ A multi-user platform.
- ❌ A general-purpose chatbot.
- ❌ An OCR or document scanning tool.

---

> **This project exists to help you learn. Break it, experiment with it, and make it your own.**
