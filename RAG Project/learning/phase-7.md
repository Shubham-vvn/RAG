# Phase 7: User Question

## What was built?
Implemented an interactive text input widget (`st.text_input`) in the Streamlit UI. The text input remains disabled when no PDF is uploaded and gets enabled once text extraction and vector database storage are complete. The app also displays the user's submitted query back in the UI.

## What problem does this solve?
RAG requires dynamic user query input to retrieve matching documents. This phase establishes the interface for capturing this query, providing feedback to the user, and managing UI state (enabling/disabling) depending on the availability of the vector index.

## Why is this phase required?
Without this phase, the user would have no way to request specific information or query the PDF. Restricting/disabling the input field when no PDF is uploaded prevents runtime errors from attempting queries on an uninitialized vector database.

## Important concepts learned
- **Disabled State**: Setting the `disabled` property on input components based on state conditions to prevent invalid actions.
- **Streamlit Execution Model**: Understanding how Streamlit runs the script from top to bottom on every user input/interaction, and how widget states (like text inputs) trigger re-runs.

## Interview questions
1. **How does Streamlit handle state updates and script re-runs when a user types in a text input?**
   - *Answer*: By default, whenever a user types in `st.text_input` and presses Enter (or clicks outside), Streamlit triggers a full script re-run from the first line to the last line. The value of `st.text_input` is preserved during the re-run and updated in the returned variable.
2. **Why is it important to disable the question input widget before a document is uploaded?**
   - *Answer*: To prevent the user from performing queries before the vector index is initialized. Doing so would lead to `NameError`, `AttributeError`, or `NoneType` exceptions in downstream retrieval code. Disabling elements is a critical design practice to enforce the correct pipeline sequence.
3. **What is the difference between a conversational chat input and a standard text input widget?**
   - *Answer*: `st.chat_input` renders a message input box pinned to the bottom of the screen, designed for chat-like interfaces. `st.text_input` renders a standard form input box inline within the application flow. Both capture string inputs, but they support different user experiences.

## Common mistakes
- **Allowing empty submissions**: Failing to check if `question` is empty or whitespace-only before proceeding to query the database.
- **Placing input inside conditional blocks**: If the input is fully nested inside `if uploaded_file is not None:`, the widget completely disappears when the file state resets. Placing the input outside and disabling it preserves visual layout consistency.

## Key takeaway
An intuitive, state-controlled interface prevents user confusion and runtime errors by ensuring query inputs are only accepted when the application is fully ready to process them.
