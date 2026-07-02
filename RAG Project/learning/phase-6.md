# Phase 6: Vector Database (ChromaDB)

## What was built?
An in-memory ChromaDB vector database integrated into the Streamlit application to store the generated text chunks and their metadata (source document name and chunk index).

## What problem does this solve?
RAG requires matching a user's question with the most relevant passages of a document. A standard keyword search (like CTRL+F) fails to capture semantic meaning. A vector database index enables fast, semantic similarity searches across large volumes of high-dimensional chunk embeddings.

## Why is this phase required?
Without a vector store, we would have to calculate cosine similarity manually across every single chunk vector on every query, which is computationally inefficient and does not scale. A vector database provides index-optimized similarity search and manages the association between text chunks, embeddings, and metadata.

## Important concepts learned
- **Vector Database**: A specialized database designed to store, manage, and query high-dimensional vector embeddings.
- **In-Memory Storage**: Storing the index in RAM for fast, transient operations, suitable for single-document workflows.
- **Metadata**: Supplementary info (e.g. chunk index, page number, file source) stored alongside vectors to enable filtering, referencing, or contextualizing retrieved results.

## Interview questions
1. **What is a vector database and why is it preferred over relational databases for RAG?**
   - *Answer*: RAG relies on semantic search (finding documents with similar meanings). Relational databases search by exact matching or string containment, which fails to capture synonyms or contextual meaning. Vector databases are optimized to store high-dimensional embeddings and perform nearest-neighbor algorithms (like HNSW) to search by distance (e.g., Cosine Distance) extremely fast.
2. **What does the metadata in a vector database do?**
   - *Answer*: Metadata allows filtering queries (e.g., only search documents uploaded by a specific user or matching a specific tag) and provides reference information when generating answers (such as displaying the chunk's source filename or page number to the user).
3. **What is the difference between persistent and in-memory vector databases?**
   - *Answer*: Persistent vector databases write indexes to disk, maintaining state across restarts or serving multiple requests. In-memory stores exist only in RAM and are wiped when the process restarts. In-memory is ideal for lightweight single-session tasks, while persistent is necessary for multi-document production systems.

## Common mistakes
- **Not syncing metadata**: Forgetting to add chunk indexes or document names makes it impossible to cite references or audit why a particular answer was generated.
- **Incompatible embedding dimensions**: Re-indexing with a different model without wiping the collection, causing embedding dimension mismatches.
- **Duplicate index ingestion**: Inserting the same document multiple times without clearing existing collection IDs.

## Key takeaway
A vector database is the search engine of the RAG pipeline. It indexes text chunks by their semantic representation (embeddings) so they can be retrieved instantly using similarity metrics.
