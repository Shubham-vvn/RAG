# Phase 8: Question Embedding

## What was built?
Integrated question embedding generation by vectorizing user-entered text queries through the same BGE embedding model (`BAAI/bge-small-en-v1.5`) using the `embed_query` method. Created an expandable UI debug section displaying the resulting question vector dimension and a preview of the first 5 values.

## What problem does this solve?
RAG operates by calculating the semantic similarity between two vectors (the user query vector and the document chunk vectors). To perform this comparison, the text query must first be projected into the same high-dimensional coordinate space as the document chunks.

## Why is this phase required?
Without question embedding, we cannot run mathematical distance calculations (like Cosine Similarity) between the user's question and the indexed document chunks. It connects the user's input to the retrieval stage.

## Important concepts learned
- **Dimensional Alignment**: Both document chunks and user queries must be encoded by the *exact same model* (down to the configuration and version) to ensure they share the same vector space coordinates and features.
- **`embed_query` vs. `embed_documents`**: `embed_query` encodes a single query string, while `embed_documents` encodes a list of documents. Depending on the model, queries and documents might be processed with slightly different prefix instructions (e.g. BGE adds search prefixes to queries for better retrieval).

## Interview questions
1. **Why must the user's question be embedded using the exact same model as the text chunks?**
   - *Answer*: If different embedding models are used, the vector dimensions might not match (e.g., 384 vs. 1536), and even if they do, the semantic features mapped to each coordinate index will be completely different. It would be like trying to find a word in an English dictionary using a search index written in Chinese.
2. **What is the difference between `embed_query` and `embed_documents` in LangChain?**
   - *Answer*: In many models (like BGE or Instructor), query embeddings and document embeddings require different prompting structures or instructions. For example, queries are prefixed with "Represent this sentence for searching relevant passages:" to tune the vectors for retrieval tasks, whereas document chunks are embedded as-is. `embed_query` handles these query-specific prefixes automatically.
3. **What is a high-dimensional vector space?**
   - *Answer*: It is a mathematical coordinate space with a high number of directions/axes (e.g., 384 or 1536). Each axis represents abstract semantic features learned by the model during training. Every text chunk and query is mapped to a single point in this space.

## Common mistakes
- **Using a different model or version**: Using a different model size (e.g., `bge-large-en-v1.5` for query and `bge-small-en-v1.5` for chunks) which will throw a dimension mismatch error during retrieval.
- **Embedding the query as a document list**: Using `embed_documents([query])` instead of `embed_query(query)`, which may bypass the query-specific search instructions.

## Key takeaway
Query embedding maps the user's real-time question into the same mathematical search space as the document index, enabling semantic comparison.
