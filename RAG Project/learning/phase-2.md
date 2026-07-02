# Phase 2: PDF Loading

## What was built?
Implemented the PDF upload UI component using Streamlit's `st.file_uploader` to accept, restrict, and display basic metadata (name, size in KB) of uploaded PDF files.

## What problem does this solve?
RAG requires an external source of information. The upload interface allows users to feed private documents (PDFs) directly to the system dynamically, rather than hardcoding static text resources inside the application.

## Why is this phase required?
Without file ingestion, there is no way for a user to query their own documents. PDF loading establishes the input boundary of the RAG document-processing pipeline.

## Important concepts learned
- **File Uploader Widget**: Streamlit's interface to handle secure file uploads from a browser client.
- **File Validation**: Limiting acceptable file extensions (e.g., `type=["pdf"]`) to ensure the downstream extraction tools do not crash on unsupported binary formats.
- **Client-Server Upload Boundary**: How uploaded files are buffered in memory (or temporary files) and exposed as file-like objects in Python.

## Interview questions
1. **How do you restrict file uploads to only PDF documents in Streamlit?**
   - *Answer*: By passing the `type=["pdf"]` parameter to the `st.file_uploader()` function. This forces the browser file chooser to disable non-PDF options and validates the file extension upon upload.
2. **What happens to the uploaded file on the server-side when a user uploads it through the UI?**
   - *Answer*: Streamlit uploads the file and wraps it in a BytesIO or temporary file wrapper, making it accessible as a file-like byte buffer. The stream can be read directly by PDF readers without saving the file to the local disk.
3. **Why do we display file metadata like name and size after uploading?**
   - *Answer*: It provides immediate visual validation and confirmation to the user that their file was successfully received, parsed, and is ready for indexing, which enhances user experience (UX).

## Common mistakes
- **No file type restrictions**: Allowing any file type (like `.jpg` or `.txt`) to be uploaded, which will crash the downstream text extraction parser.
- **Ignoring size limits**: Allowing massive files that exceed RAM capacity, causing out-of-memory (OOM) errors in resource-constrained environments.

## Key takeaway
Dynamic file uploading is the starting point of the RAG data ingestion pipeline, requiring strict type control and clear user feedback.
