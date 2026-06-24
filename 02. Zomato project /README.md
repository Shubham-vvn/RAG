# ZomaAI — AI-Powered Restaurant Recommendation System

> An intelligent restaurant recommendation app powered by **Groq LLM** and the **Zomato dataset**.

---

## 🚀 Quick Start

### 1. Get the Dataset
```bash
pip install datasets pandas
python preprocess_dataset.py
```
This downloads and preprocesses the Zomato dataset from HuggingFace into `data/zomato_dataset.json`.

### 2. Add Your Groq API Key
1. Get a free key from [console.groq.com](https://console.groq.com)
2. Open `js/config.js` and replace `YOUR_GROQ_API_KEY_HERE` with your key

### 3. Run the App
Open `index.html` in your browser (use a local server for ES module support):
```bash
# Python
python -m http.server 8080

# Node.js
npx serve .
```
Then visit `http://localhost:8080`

---

## 📂 Project Structure

```
02. Zomato project/
├── index.html                  # Main UI entry point
├── css/styles.css              # Design system & dark mode
├── js/
│   ├── app.js                  # App coordinator
│   ├── filters.js              # Dataset filtering engine
│   ├── recommendations.js      # Groq LLM integration & card renderer
│   └── config.js               # ⚠️ API key (gitignored — never commit)
├── data/zomato_dataset.json    # Preprocessed dataset (run preprocess_dataset.py)
├── preprocess_dataset.py       # Dataset download + preprocessing script
└── docs/
    ├── context.md
    ├── architecture.md
    ├── implementation_plan.md
    └── edge-cases.md
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, Vanilla CSS, Vanilla JS (ES Modules) |
| Dataset | HuggingFace — `ManikaSaini/zomato-restaurant-recommendation` |
| LLM | [Groq](https://groq.com) — `llama3-8b-8192` |
| Preprocessing | Python, `datasets`, `pandas` |

---

## ⚠️ Security Note

`js/config.js` is listed in `.gitignore` and must **never** be committed.  
If your API key is accidentally pushed, rotate it immediately at [console.groq.com](https://console.groq.com).
