# Phase 4: Chunking

## What was built?
Configured LangChain's `RecursiveCharacterTextSplitter` to split extracted text into smaller, overlapping chunks (`chunk_size=500`, `chunk_overlap=50`) and displayed a preview of the first three chunks in the UI.

## What problem does this solve?
Embedding models and LLMs have strict context window limits. Feeding an entire 50-page PDF to an embedding model is either impossible or dilutes the embedding vector's representation. Chunking cuts documents into bite-sized semantic pieces, enabling high-precision retrieval.

## Why is this phase required?
Without chunking, RAG cannot retrieve specific paragraphs. The entire document would have to be retrieved, quickly hitting LLM context size limitations or introducing noise that impairs response generation quality (the "lost in the middle" effect).

## Important concepts learned
- **Recursive Character Splitting**: Splitting text based on a hierarchy of characters (like `\n\n`, `\n`, ` `, and `""`) to keep paragraphs and sentences together.
- **Chunk Size**: The maximum number of characters/tokens in a single chunk.
- **Chunk Overlap**: The number of characters shared between adjacent chunks, preserving context and meaning that might otherwise get cut off at boundaries.

## Interview questions
1. **How does `RecursiveCharacterTextSplitter` decide where to split text?**
   - *Answer*: It looks at a prioritized list of separators (typically `["\n\n", "\n", " ", ""]`). It tries to split by the first separator. If the resulting pieces are still larger than the target chunk size, it recursively splits those pieces by the next separator down the list, and so on.
2. **Why do we need chunk overlap?**
   - *Answer*: If a sentence or idea is split exactly at a boundary, its context is lost. Overlap ensures that adjacent chunks share a prefix/suffix, allowing semantic search to find either chunk without losing context.
3. **What are the trade-offs of choosing a very large vs. very small chunk size?**
   - *Answer*: Very large chunks preserve broader context but dilute embedding specificity and consume more LLM tokens. Very small chunks are highly specific but can lose context needed to generate complete, cohesive answers.

## Common mistakes
- **Setting overlap to 0**: Severing sentences in half, causing context fragmentation.
- **Hardcoding arbitrary separators**: Using splitters that don't respect semantic boundaries (like hard token limits that cut words in half).

## Key takeaway
Chunking determines the granularity of your knowledge base. Selecting the right chunk size and overlap balances semantic coherence with retrieval precision.
