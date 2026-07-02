# Phase 3: Text Extraction

## What was built?
Integrated `PyPDF2` (`PdfReader`) to read pages from the uploaded file, extract plain text from each page, aggregate the text into a single string, and display it in an expandable section with character counts.

## What problem does this solve?
PDFs are binary files optimized for visual presentation, not machine reading. Raw binary bytes cannot be fed directly to embedding models or LLMs. Text extraction extracts the raw string representation of the PDF content for downstream text processing.

## Why is this phase required?
Without text extraction, the system cannot parse the contents of the document. RAG cannot run unless the source document is converted into structured plain text strings.

## Important concepts learned
- **PDF Layout vs. Plain Text**: PDFs store elements with absolute page positions. Parsers must scan these positions and reconstruct them into read order.
- **Page Iteration**: Reading page-by-page to reconstruct the continuous document.
- **Text Aggregation**: Accumulating strings from distinct pages into a single cohesive body of text.

## Interview questions
1. **How does `PyPDF2` extract text from a PDF file?**
   - *Answer*: It reads the PDF cross-reference table, locates page objects, accesses the stream contents of each page, decodes text operators (like Tj or TJ), maps them using the document's font encodings, and returns a plain text string.
2. **What are the limitations of plain-text PDF extraction?**
   - *Answer*: It cannot extract text from scanned images (which require OCR engines like Tesseract), and it often struggles with multi-column layouts, tables, math symbols, and header/footer separation, which can result in jumbled text.
3. **Why do we display the character count of extracted text?**
   - *Answer*: Character count is a simple heuristic to verify text extraction succeeded. If the count is 0, it tells the user or developer that the PDF might be scanned/encrypted, requiring different ingestion methods.

## Common mistakes
- **Not checking for empty text**: Not handling empty pages or scanned PDFs that yield no extracted characters, causing crashes down the line.
- **Ignoring encoding issues**: Failing to handle special characters or ligature mappings, which result in gibberish text representation.

## Key takeaway
Text extraction translates absolute layout elements in a PDF into clean plain text streams that machine learning models can read and represent.
