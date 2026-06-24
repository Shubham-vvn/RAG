# Architecture: AI-Powered Restaurant Recommendation System

> Derived from: `context.md` → `problemStatement.txt`  
> Last Updated: 2026-06-19

---

## 1. 🗺️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser / UI)                        │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              User Preference Input Form                     │   │
│   │  [ Location ] [ Budget ] [ Cuisine ] [ Rating ] [ Others ] │   │
│   └────────────────────────┬────────────────────────────────────┘   │
│                            │  User submits preferences              │
│   ┌────────────────────────▼────────────────────────────────────┐   │
│   │                   app.js (Coordinator)                      │   │
│   │   Orchestrates: Input → Filter → Prompt → LLM → Display    │   │
│   └──────┬─────────────────────────────────────────────┬───────┘   │
│          │                                             │           │
│   ┌──────▼──────────┐                    ┌────────────▼─────────┐  │
│   │   filters.js    │                    │  recommendations.js  │  │
│   │ Dataset Filter  │                    │  LLM API + Renderer  │  │
│   └──────┬──────────┘                    └────────────┬─────────┘  │
│          │                                            │            │
└──────────┼────────────────────────────────────────────┼────────────┘
           │                                            │
           ▼                                            ▼
┌─────────────────────┐                  ┌──────────────────────────┐
│  Zomato Dataset     │                  │       Groq API           │
│  (HuggingFace /     │                  │  (llama3-8b-8192 /       │
│   Local JSON)       │                  │   mixtral-8x7b-32768)    │
└─────────────────────┘                  └──────────────────────────┘
```

---

## 2. 🧱 Component Breakdown

### 2.1 Frontend Layer (`index.html`, `css/styles.css`, `js/app.js`)

| Component | Responsibility |
|---|---|
| **Input Form** | Collects location, budget, cuisine, rating, and extra preferences |
| **app.js** | Central coordinator — calls filters, builds prompt, calls LLM API, renders results |
| **Recommendation Cards** | Displays Name, Cuisine, Rating, Cost, AI Explanation per restaurant |
| **styles.css** | Design tokens, dark mode, responsive grid, animation utilities |

---

### 2.2 Data Layer (`data/zomato_dataset.json`)

| Aspect | Detail |
|---|---|
| **Source** | Hugging Face: [`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) |
| **Format** | JSON (loaded locally or fetched via HuggingFace Datasets API) |
| **Key Fields** | `name`, `location`, `cuisines`, `cost_for_two`, `aggregate_rating`, `votes`, `highlights` |
| **Preprocessing** | Normalize budget bands (Low/Medium/High), standardize cuisine labels, filter nulls |

---

### 2.3 Filtering Layer (`js/filters.js`)

Responsible for narrowing the full dataset to a relevant candidate list before passing to the LLM.

**Filter Logic:**

```
Input: { location, budget, cuisine, minRating, extras }

Step 1: location filter    → match city/area substring
Step 2: budget filter      → map (Low → ₹0–300, Medium → ₹300–700, High → ₹700+)
Step 3: cuisine filter     → case-insensitive includes match
Step 4: rating filter      → aggregate_rating >= minRating
Step 5: sort by votes DESC → surface popular results first
Step 6: slice top N (e.g. 10–15) → pass to Integration Layer
```

---

### 2.4 Integration / Prompt Layer (`js/recommendations.js`)

Transforms filtered restaurant data into a structured LLM prompt.

**Prompt Template:**

```
You are a restaurant recommendation assistant.
A user is looking for a restaurant with the following preferences:
  - Location: {location}
  - Budget: {budget}
  - Cuisine: {cuisine}
  - Minimum Rating: {minRating}
  - Additional Preferences: {extras}

Here are {N} candidate restaurants from the Zomato dataset:
{restaurant_list_as_structured_text}

Please:
1. Rank the top 3–5 restaurants most suitable for this user.
2. For each, explain in 2–3 sentences why it is a good fit.
3. Optionally provide a brief overall summary.

Output in valid JSON with the following schema:
[
  {
    "rank": 1,
    "name": "...",
    "cuisine": "...",
    "rating": ...,
    "cost_for_two": "...",
    "explanation": "..."
  }
]
```

---

### 2.5 LLM API Layer (`js/recommendations.js`)

| Attribute | Detail |
|---|---|
| **Provider** | [Groq](https://console.groq.com) — ultra-low latency LLM inference |
| **Primary Model** | `llama3-8b-8192` (fast, cost-free on free tier) |
| **Alternate Model** | `mixtral-8x7b-32768` (larger context window, higher quality) |
| **API Endpoint** | `https://api.groq.com/openai/v1/chat/completions` |
| **Communication** | `fetch()` via REST — OpenAI-compatible JSON body with `messages` array |
| **Auth** | Groq API Key stored in `config.js` (excluded from version control via `.gitignore`) |
| **Error Handling** | Retry on timeout, display fallback message on failure |
| **Response Parsing** | `JSON.parse()` on LLM output → render cards |

---

### 2.6 Output / Display Layer

Each recommendation card renders:

```
┌─────────────────────────────────────────────────┐
│  🥇  The Spice Route                            │
│  🥘 North Indian  |  ⭐ 4.5  |  💰 ₹600 for 2  │
│                                                 │
│  🤖 "A perfect match for your love of North     │
│  Indian food in Delhi. Known for its rich       │
│  curries and cozy family ambiance."             │
└─────────────────────────────────────────────────┘
```

---

## 3. 🔁 End-to-End Data Flow

```
[User fills form]
       │
       ▼
[app.js reads input values]
       │
       ▼
[filters.js filters zomato_dataset.json]
  → Apply location, budget, cuisine, rating filters
  → Return top N candidate restaurants
       │
       ▼
[recommendations.js builds LLM prompt]
  → Inject user prefs + candidate restaurant list
       │
       ▼
[Groq API call (fetch POST)]
  → Send prompt to Groq (`llama3-8b-8192`)
  → Receive JSON ranked recommendations
       │
       ▼
[recommendations.js parses response]
  → Validate JSON structure
  → Trigger card rendering
       │
       ▼
[app.js renders result cards in UI]
  → Name, Cuisine, Rating, Cost, AI Explanation
```

---

## 4. 📂 Full Project File Structure

```text
02. Zomato project/
├── index.html                   # Entry point — form + results layout
├── context.md                   # Project context & scope
├── architecture.md              # This file — system design document
├── problemStatement.txt         # Raw problem statement
│
├── css/
│   └── styles.css               # Global styles, design tokens, dark mode
│
├── js/
│   ├── app.js                   # Coordinator: init, event listeners, render
│   ├── filters.js               # Dataset filtering logic
│   ├── recommendations.js       # Prompt building, LLM API call, response parse
│   └── config.js                # API keys & config constants (gitignored)
│
├── data/
│   └── zomato_dataset.json      # Pre-processed local dataset snapshot
│
└── assets/
    ├── logo.svg                 # App logo
    └── icons/                   # UI icons (cuisine, rating, cost)
```

---

## 5. 🔐 Security & Configuration

| Concern | Approach |
|---|---|
| **API Key Storage** | `config.js` — never committed to version control |
| **`.gitignore`** | Add `js/config.js` and `data/*.json` if sensitive |
| **CORS** | LLM API calls made client-side; use a lightweight proxy if CORS blocks arise |
| **Rate Limiting** | Debounce user submissions; show loading state to prevent repeated calls |

---

## 6. 🚧 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Client-side first** | Simpler setup — no backend server required for MVP |
| **Pre-filter before LLM** | Reduces token usage and improves response quality |
| **LLM output as JSON** | Reliable structured parsing for card rendering |
| **Top 3–5 recommendations** | Avoids overwhelming the user; LLM has enough context to rank meaningfully |
| **LocalStorage for state** | Persist last search preferences across page reloads without a backend |

---

## 7. ✅ Acceptance Criteria

- [ ] User can input preferences and submit successfully
- [ ] Dataset is filtered to relevant candidates before calling the LLM
- [ ] LLM returns ranked JSON recommendations with explanations
- [ ] Result cards display all required fields (name, cuisine, rating, cost, explanation)
- [ ] App handles API errors gracefully (retry / fallback message)
- [ ] UI is responsive across desktop and mobile screen sizes
- [ ] Dark mode and animations are implemented
