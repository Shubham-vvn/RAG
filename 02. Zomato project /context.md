# Project Context: AI-Powered Restaurant Recommendation System (Zomato Use Case)

> Source: `problemStatement.txt`  
> Last Updated: 2026-06-19

---

## 📌 Problem Statement

You are tasked with building an **AI-powered restaurant recommendation service** inspired by Zomato. The system should intelligently suggest restaurants based on user preferences by combining **structured data** with a **Large Language Model (LLM)**.

---

## 🎯 Objective

Design and implement an application that:

- Takes **user preferences** (such as location, budget, cuisine, and ratings)
- Uses a **real-world dataset** of restaurants
- Leverages an **LLM** to generate personalized, human-like recommendations
- Displays **clear and useful results** to the user

---

## 🔄 System Workflow

### 1. Data Ingestion
- Load and preprocess the **Zomato dataset** from Hugging Face:  
  👉 [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- Extract relevant fields:
  - Restaurant Name
  - Location
  - Cuisine
  - Cost
  - Rating
  - (and other applicable fields)

---

### 2. User Input
Collect user preferences via an interface:

| Preference | Example Values |
|---|---|
| Location | Delhi, Bangalore |
| Budget | Low / Medium / High |
| Cuisine | Italian, Chinese, Indian, etc. |
| Minimum Rating | e.g., 4.0+ |
| Additional Preferences | Family-friendly, Quick service, etc. |

---

### 3. Integration Layer
- **Filter** the dataset based on user inputs
- **Prepare** a structured summary of relevant restaurants
- **Design an LLM prompt** that enables the model to reason and rank options intelligently

---

### 4. Recommendation Engine (LLM)
Use the LLM to:
- **Rank** restaurants based on suitability
- **Explain** why each recommendation fits the user's preferences
- (Optional) **Summarize** the overall choices in a conversational manner

---

### 5. Output Display
Present **top recommendations** in a user-friendly format:

- 🍽️ **Restaurant Name**
- 🥘 **Cuisine**
- ⭐ **Rating**
- 💰 **Estimated Cost**
- 🤖 **AI-generated explanation** (why this restaurant was recommended)

---

## 🛠️ Tech Stack (Proposed)

| Layer | Technology |
|---|---|
| Dataset | Hugging Face — `ManikaSaini/zomato-restaurant-recommendation` |
| LLM Integration | [Groq](https://console.groq.com) — `llama3-8b-8192` / `mixtral-8x7b-32768` |
| Backend / Logic | Python (pandas for data, LangChain or direct API calls) |
| Frontend UI | HTML, CSS, Vanilla JavaScript |
| State / Persistence | LocalStorage / Session |

---

## 📂 Proposed Project Structure

```text
02. Zomato project/
├── context.md                  # Project context & scope (this file)
├── problemStatement.txt        # Raw problem statement
├── index.html                  # Main UI entry point
├── css/
│   └── styles.css              # Styling & design tokens
├── js/
│   ├── app.js                  # Main UI coordinator
│   ├── filters.js              # User preference filtering logic
│   └── recommendations.js     # LLM API call & response rendering
├── data/
│   └── zomato_dataset.json     # Local snapshot of the dataset (optional)
└── assets/                     # Icons, images, logos
```

---

## ✅ Next Steps / Milestones

- [ ] Download & explore the Zomato dataset from Hugging Face
- [ ] Set up data ingestion and preprocessing pipeline
- [ ] Build user preference input UI (location, budget, cuisine, rating)
- [ ] Implement backend filtering logic
- [ ] Design and test LLM prompt for restaurant ranking & explanation
- [ ] Connect LLM API and handle responses
- [ ] Render top recommendations in a clean, formatted UI
- [ ] Polish UI with animations, dark mode, and responsive layout
