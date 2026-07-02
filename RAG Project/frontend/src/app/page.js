"use client";

import { useState, useEffect } from "react";
import styles from "./page.module.css";

export default function Home() {
  // Ingested Sources Dashboard State
  const [sources, setSources] = useState({});
  
  // File Uploader States
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState([]); // Array of { name, status: 'pending'|'loading'|'success'|'error', message }
  const [isUploading, setIsUploading] = useState(false);
  
  // Link Ingestor States
  const [linkUrl, setLinkUrl] = useState("");
  const [isIngestingLink, setIsIngestingLink] = useState(false);
  const [linkStatus, setLinkStatus] = useState(null); // { status: 'success'|'error', message }

  // Question/Answer States
  const [question, setQuestion] = useState("");
  const [isAnswering, setIsAnswering] = useState(false);
  const [answer, setAnswer] = useState("");
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  // Fetch active sources on mount
  useEffect(() => {
    fetchSources();
  }, []);

  const fetchSources = async () => {
    try {
      const response = await fetch("http://localhost:8000/sources");
      if (response.ok) {
        const data = await response.json();
        setSources(data.sources || {});
      }
    } catch (err) {
      console.error("Error fetching indexed sources:", err);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(files);
      setUploadProgress(files.map(f => ({ name: f.name, status: "pending", message: "" })));
    }
  };

  const handleIngestContent = async (e) => {
    e.preventDefault();

    const hasFiles = selectedFiles.length > 0;
    const hasLink = !!linkUrl.trim();
    if (!hasFiles && !hasLink) return;

    let uploadPromise = Promise.resolve();
    let linkPromise = Promise.resolve();

    if (hasFiles) {
      uploadPromise = (async () => {
        setIsUploading(true);
        const newProgress = [...uploadProgress];

        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          newProgress[i].status = "loading";
          setUploadProgress([...newProgress]);

          const formData = new FormData();
          formData.append("file", file);

          try {
            const response = await fetch("http://localhost:8000/upload", {
              method: "POST",
              body: formData,
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.detail || "Failed to index file.");
            }

            newProgress[i].status = "success";
            newProgress[i].message = "Indexed successfully";
          } catch (err) {
            newProgress[i].status = "error";
            newProgress[i].message = err.message || "Failed to process file.";
          }
          setUploadProgress([...newProgress]);
        }

        setIsUploading(false);
        setSelectedFiles([]);
      })();
    }

    if (hasLink) {
      linkPromise = (async () => {
        setIsIngestingLink(true);
        setLinkStatus(null);

        try {
          const response = await fetch("http://localhost:8000/upload-link", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: linkUrl }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Failed to crawl web URL.");
          }

          setLinkStatus({ status: "success", message: "Web URL content successfully indexed." });
          setLinkUrl("");
        } catch (err) {
          setLinkStatus({ status: "error", message: err.message || "Failed to crawl URL." });
        } finally {
          setIsIngestingLink(false);
        }
      })();
    }

    await Promise.all([uploadPromise, linkPromise]);
    fetchSources(); // Refresh dashboard
  };

  const getButtonText = () => {
    if (isUploading && isIngestingLink) return "Indexing Files & Crawling Link...";
    if (isUploading) return "Indexing Files...";
    if (isIngestingLink) return "Crawling Link...";

    const hasFiles = selectedFiles.length > 0;
    const hasLink = !!linkUrl.trim();

    if (hasFiles && hasLink) return "Ingest Files & Link";
    if (hasFiles) return "Ingest Selected Files";
    if (hasLink) return "Ingest Link";
    return "Ingest Content";
  };

  const handleClearIndex = async () => {
    if (!confirm("Are you sure you want to delete all indexed documents and links?")) return;

    try {
      const response = await fetch("http://localhost:8000/clear", {
        method: "POST"
      });
      if (response.ok) {
        setSources({});
        setAnswer("");
        setRetrievedChunks([]);
        setUploadProgress([]);
        setLinkStatus(null);
        setErrorMsg("");
      }
    } catch (err) {
      console.error("Failed to clear vector store:", err);
      alert("Failed to reset database.");
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || Object.keys(sources).length === 0) return;

    setIsAnswering(true);
    setAnswer("");
    setRetrievedChunks([]);
    setErrorMsg("");

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate answer.");
      }

      const data = await response.json();
      setAnswer(data.answer);
      setRetrievedChunks(data.retrieved_chunks);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "An error occurred while answering.");
    } finally {
      setIsAnswering(false);
    }
  };

  const getSourceBadge = (filename, sourceData) => {
    if (sourceData.type === "link") return styles.badgeLink;
    const ext = filename.split(".").pop().toLowerCase();
    switch (ext) {
      case "pdf": return styles.badgePdf;
      case "docx": return styles.badgeDocx;
      case "csv": return styles.badgeCsv;
      default: return styles.badgeTxt;
    }
  };

  return (
    <main className={styles.main}>
      {/* Header Banner */}
      <header className={styles.header}>
        <div className={styles.logoArea}>
          <span className={styles.icon}>📚</span>
          <div>
            <h1>Ask My Documents</h1>
            <p className={styles.tagline}>Cross-document Grounded Intelligence via Local Vector Storage & Groq LLM</p>
          </div>
        </div>
        
        {/* Status Indicator */}
        <div className={styles.statusBadge}>
          <span className={`${styles.dot} ${Object.keys(sources).length > 0 ? styles.online : styles.offline}`}></span>
          <span>
            {Object.keys(sources).length > 0 
              ? `${Object.keys(sources).length} Source(s) Loaded` 
              : "Upload documents or paste links"
            }
          </span>
        </div>
      </header>

      {/* Sources Dashboard */}
      <section className={styles.dashboardCard}>
        <div className={styles.dashboardHeader}>
          <h2>🗂️ Active Sources Index</h2>
          {Object.keys(sources).length > 0 && (
            <button onClick={handleClearIndex} className={styles.dangerButton}>
              Reset Database Index
            </button>
          )}
        </div>
        
        {Object.keys(sources).length === 0 ? (
          <p className={styles.emptyText}>No sources currently indexed. Please upload files or add web links below.</p>
        ) : (
          <div className={styles.sourcesList}>
            {Object.entries(sources).map(([name, info]) => (
              <div key={name} className={styles.sourceItem}>
                <div className={styles.sourceMain}>
                  <span className={`${styles.badge} ${getSourceBadge(name, info)}`}>
                    {info.type === "link" ? "LINK" : name.split(".").pop().toUpperCase()}
                  </span>
                  <span className={styles.sourceName} title={name}>{name}</span>
                </div>
                <div className={styles.sourceMeta}>
                  <span>{(info.char_count / 1024).toFixed(1)} KB</span>
                  <span className={styles.metaDivider}>|</span>
                  <span>{info.chunk_count} Chunks</span>
                  {info.pages && (
                    <>
                      <span className={styles.metaDivider}>|</span>
                      <span>{info.pages} Pages</span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <div className={styles.grid}>
        {/* Ingest Section: Multiple files and URL input */}
        <section className={styles.card}>
          <h2>1. Ingest Content</h2>
          <p className={styles.description}>Add multiple documents (.pdf, .docx, .txt, .md, .csv) or paste website links to crawl and vectorize.</p>
          
          {/* Unified Ingestion Form */}
          <form onSubmit={handleIngestContent} className={styles.ingestForm}>
            {/* File Upload zone */}
            <label className={styles.dropzone}>
              <input 
                type="file" 
                accept=".pdf,.docx,.txt,.md,.csv" 
                multiple
                onChange={handleFileChange} 
                className={styles.fileInput}
              />
              <span className={styles.uploadIcon}>📂</span>
              <span className={styles.uploadText}>
                {selectedFiles.length > 0 
                  ? `${selectedFiles.length} file(s) selected` 
                  : "Click to select local files"
                }
              </span>
              <span className={styles.fileLimit}>Supports PDF, DOCX, TXT, MD, CSV</span>
            </label>

            {/* Files Progress Panel */}
            {uploadProgress.length > 0 && (
              <div className={styles.progressContainer}>
                {uploadProgress.map((prog, idx) => (
                  <div key={idx} className={styles.progressItem}>
                    <span className={styles.progName}>{prog.name}</span>
                    <span className={`${styles.progStatus} ${styles[prog.status]}`}>
                      {prog.status === "loading" && "Processing..."}
                      {prog.status === "success" && "✅ Indexed"}
                      {prog.status === "error" && "❌ Failed"}
                      {prog.status === "pending" && "Pending"}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className={styles.divider}><span>or URL reference</span></div>

            {/* URL Input */}
            <div className={styles.linkInputGroup}>
              <span className={styles.linkPrefix}>🔗</span>
              <input
                type="url"
                placeholder="https://example.com/article"
                value={linkUrl}
                onChange={(e) => setLinkUrl(e.target.value)}
                disabled={isIngestingLink}
                className={styles.linkInput}
              />
            </div>

            {linkStatus && (
              <div className={`${styles.panel} ${linkStatus.status === "success" ? styles.successPanel : styles.errorPanel}`}>
                <p>{linkStatus.message}</p>
              </div>
            )}

            <button 
              type="submit" 
              disabled={(!selectedFiles.length && !linkUrl.trim()) || isUploading || isIngestingLink}
              className={styles.primaryButton}
            >
              {getButtonText()}
            </button>
          </form>
        </section>

        {/* Query Section: Ask & Retrieve */}
        <section className={styles.card}>
          <h2>2. Grounded Multi-Source Q&A</h2>
          <p className={styles.description}>Submit questions. Answers are compiled strictly from the context blocks in your loaded sources.</p>

          <form onSubmit={handleAsk} className={styles.queryForm}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={Object.keys(sources).length === 0 ? "Please index some content sources first..." : "Ask a question across all sources..."}
              disabled={Object.keys(sources).length === 0 || isAnswering}
              className={styles.textInput}
            />
            <button 
              type="submit" 
              disabled={!question.trim() || Object.keys(sources).length === 0 || isAnswering}
              className={styles.accentButton}
            >
              {isAnswering ? "Querying..." : "Ask"}
            </button>
          </form>

          {/* Answer Area */}
          {answer && (
            <div className={styles.answerArea}>
              <h3>🤖 Generated Grounded Answer</h3>
              <p className={styles.answerText}>{answer}</p>
            </div>
          )}

          {errorMsg && (
            <div className={`${styles.panel} ${styles.errorPanel}`}>
              <p>{errorMsg}</p>
            </div>
          )}

          {isAnswering && (
            <div className={styles.loadingArea}>
              <div className={styles.spinner}></div>
              <p>Scanning vector store and synthesizing response...</p>
            </div>
          )}
        </section>
      </div>

      {/* Semantic Retrieval Inspector */}
      {retrievedChunks.length > 0 && (
        <section className={`${styles.card} ${styles.wideCard}`}>
          <h2>🔍 Semantic Retrieval Inspector</h2>
          <p className={styles.description}>Top 3 matches fetched from ChromaDB. Verify metadata origins and L2 similarity coordinates below.</p>
          
          <div className={styles.chunksGrid}>
            {retrievedChunks.map((chunk) => (
              <div key={chunk.index} className={styles.chunkCard}>
                <div className={styles.chunkHeader}>
                  <h3>📄 Source Chunk {chunk.index}</h3>
                  <span className={styles.scoreBadge}>L2 Dist: {chunk.score.toFixed(4)}</span>
                </div>
                <div className={styles.chunkSourceLine}>
                  <strong>Origin:</strong> <code className={styles.sourceCode} title={chunk.metadata.source}>{chunk.metadata.source}</code>
                </div>
                <div className={styles.explanationBox}>
                  <strong>Why retrieved:</strong> {chunk.explanation}
                </div>
                <div className={styles.chunkContent}>
                  <p>{chunk.text}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
