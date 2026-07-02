# Phase 5: Embedding Generation (BGE)

## What was built?
Loaded the BGE embedding model (`BAAI/bge-small-en-v1.5`) via `langchain-huggingface` to convert plain text chunks into vector embeddings. Created a debug script to save generated vectors to `debug/embeddings.json` for learning.

## What problem does this solve?
Computers cannot compare text string meanings directly. Embeddings convert text into a list of floating-point numbers (a vector) where geometrically close vectors represent semantically similar concepts (e.g., "king" and "queen" will be close, while "king" and "banana" will be far).

## Why is this phase required?
Embedding generation translates human-readable text into a format suitable for similarity calculation. Semantic search in RAG cannot function without mapping chunks and queries into a shared vector space.

## Important concepts learned
- **Embedding Model**: A neural network trained to project tokens or sentences into a dense, low-dimensional continuous vector space.
- **Embedding Dimension**: The length of the output vector (e.g., 384 for `bge-small-en-v1.5`), representing different features of the semantic meaning.
- **Dense Representation**: Packing complex semantic features into a dense float array, as opposed to sparse representations like TF-IDF.

## Interview questions
1. **What is the difference between a dense embedding model and a sparse retrieval method like BM25?**
   - *Answer*: Sparse retrieval (BM25) matches exact words or keywords and scoring relies on frequency. Dense embeddings capture semantic relationships and synonyms, mapping words with similar meanings to close vectors even if they share zero characters.
2. **Why does `bge-small-en-v1.5` have a dimension of 384, and how does this affect performance?**
   - *Answer*: 384 is the hidden size of the transformer's output representation. A smaller dimension reduces memory footprints, disk usage, and comparison time, making it faster to run locally on CPUs, though it might capture slightly fewer complex semantic relationships than a 1536-dimension model.
3. **What is cosine similarity and how is it used in semantic matching?**
   - *Answer*: Cosine similarity measures the cosine of the angle between two vectors. It ranges from -1 to 1. In text search, it calculates how close the query vector's direction is to the chunk vectors' directions, indicating how closely related the topics are regardless of chunk length.

## Common mistakes
- **Mixing models**: Ingesting documents with one embedding model and querying with a different one. The dimensions and vector spaces will not align, rendering results meaningless.
- **Ignoring hardware limits**: Running heavy embedding models locally without checking for GPU availability or RAM capacity, leading to slow processing times.

## Key takeaway
Embeddings turn plain text into geometric points in a high-dimensional space, translating semantic meaning into mathematical distance.
