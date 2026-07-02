# Phase 9: Retrieval

## What was built?
Implemented semantic retrieval querying from ChromaDB using `similarity_search_with_score` (K=3). Resolved the corresponding document IDs by indexing metadata (`chunk_index`), added a dynamic text analyzer that highlights keyword overlap, and rendered the results inside an expandable debug container.

## What problem does this solve?
RAG needs to fetch only the relevant portions of the document to feed as context into the LLM. Retrieval filters the massive amount of content from a PDF down to the top-3 most semantically relevant paragraphs, keeping token usage low and focus high.

## Why is this phase required?
Without retrieval, we would be forced to send the entire PDF text to the LLM on every query, which is extremely expensive, slow, and often hits context window limits. Retrieval acts as a high-precision search filter for the generation stage.

## Important concepts learned
- **K Nearest Neighbors (K-NN)**: Finding the K points in vector space that are closest to a target query vector.
- **Distance Metric (L2 Distance)**: The straight-line Euclidean distance between two vectors. A lower L2 score represents vectors that are closer together (i.e. more semantically similar).
- **Similarity Search with Score**: Returning both the relevant text chunk and its numeric distance metric, which indicates search confidence.

## Interview questions
1. **What is the difference between Euclidean Distance (L2) and Cosine Similarity?**
   - *Answer*: Euclidean Distance measures the geometric distance between the endpoints of two vectors (taking scale and magnitude into account). Cosine Similarity measures the angle between the two vectors (focusing purely on direction). For normalized embeddings, the two metrics are mathematically equivalent (minimizing L2 distance is the same as maximizing Cosine similarity).
2. **What does the parameter `K` represent in similarity search, and how do we choose it?**
   - *Answer*: `K` is the number of closest documents to retrieve. A larger `K` provides more context to the LLM but increases token cost and noise. A smaller `K` is faster and cheaper but may miss necessary information. Usually, `K` is chosen between 3 and 10 depending on chunk size and context window limits.
3. **What is "Lost in the Middle" and how does retrieval relate to it?**
   - *Answer*: Research shows that LLMs are best at processing information at the very beginning or end of their input context, and often ignore or miss facts located in the middle of long prompts. Keeping `K` reasonably small and sorting retrieved chunks by highest relevance ensures the LLM handles crucial facts optimally.

## Common mistakes
- **Ignoring distance scores**: Not showing or filtering by similarity scores, which could lead to retrieving completely irrelevant chunks if the PDF doesn't contain the answer.
- **Confusing distance with similarity**: Assuming a higher score is better. In Chroma's default configuration, a lower distance score means higher similarity.

## Key takeaway
Retrieval selects the most relevant segments of context from a database to ground the LLM's answers in specific facts rather than generalized training weights.
