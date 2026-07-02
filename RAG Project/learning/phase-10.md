# Phase 10: Answer Generation

## What was built?
Integrated the Groq API utilizing `ChatGroq` (`llama-3.1-8b-instant`) to feed the user query and the retrieved context chunks into a grounded system prompt. The model processes this combination and streams/displays natural language answers back to the user.

## What problem does this solve?
RAG converts retrieved chunks of text back into readable answers that solve the user's specific query. The LLM translates raw document passages into direct, concise, and context-grounded responses rather than forcing the user to manually read the retrieved sections.

## Why is this phase required?
Without answer generation, the system would only be a search engine (like Google) displaying matching document blocks. Answer generation completes the "generation" part of Retrieval-Augmented Generation (RAG).

## Important concepts learned
- **Grounded Generation (System Prompting)**: Explicitly telling the LLM to *only* answer using the provided context, which forces it to suppress its training knowledge and hallucinations.
- **Temperature Configuration**: Setting the temperature parameter to `0.0` to minimize creativity and maximize factuality and determinism.
- **Context Injection**: Structuring the prompt payload to separate System instructions, retrieved Context blocks, and the User's Query.

## Interview questions
1. **What is a "hallucination" in LLMs and how does RAG mitigate it?**
   - *Answer*: Hallucination occurs when an LLM generates factual errors or confidently made-up statements because it lacks access to source facts or tries to predict the next word based on generalized training patterns. RAG mitigates this by injecting relevant source documents into the context window and instructing the LLM to answer *only* from that reference text.
2. **Why do we set the LLM's `temperature` to `0.0` in a RAG pipeline?**
   - *Answer*: Temperature controls the randomness/creativity of the generated text. In RAG pipelines, we want the LLM to behave like a strict search index summarizer, not a creative writer. A temperature of `0.0` makes the output deterministic and mathematically ties the generation directly to the source documents.
3. **What happens if the context provided does not contain the answer to the user's question?**
   - *Answer*: The system prompt must instruct the model to explicitly state "I don't have enough information to answer this question." If this instruction is absent, the LLM will fall back on its pre-trained general knowledge, violating the grounding guarantee and introducing hallucinations.

## Common mistakes
- **Vague system prompts**: Failing to restrict the LLM to the provided context, letting it answer using general training weights.
- **Hardcoding keys in scripts**: Exposing the API key in the source code instead of loading it securely using `os.getenv` or `python-dotenv`.
- **Not handling API failures**: Failing to catch exceptions (network dropouts, rate limits, invalid keys) gracefully, crashing the UI.

## Key takeaway
Answer generation transforms raw database retrieval results into human-friendly, factually-grounded answers by feeding them into a temperature-controlled LLM.
