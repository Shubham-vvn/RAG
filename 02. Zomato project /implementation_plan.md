# Phase-Wise Implementation Plan
# AI-Powered Restaurant Recommendation System (Zomato Use Case)

> Based on: `architecture.md` + `context.md`  
> Last Updated: 2026-06-19

---

## 📋 Overview

| Phase | Name | Focus Area | Estimated Effort |
|---|---|---|---|
| **Phase 1** | Project Setup & Data Ingestion | Foundation, dataset, file structure | Day 1 |
| **Phase 2** | UI & Input Layer | HTML structure, CSS design system, form | Day 2 |
| **Phase 3** | Filtering Engine | Dataset filtering logic in `filters.js` | Day 3 |
| **Phase 4** | Groq LLM Integration | Prompt design, API call, response parsing | Day 4 |
| **Phase 5** | Output Rendering | Recommendation cards, results display | Day 5 |
| **Phase 6** | Polish & Error Handling | Animations, dark mode, edge cases | Day 6 |
| **Phase 7** | Testing & Wrap-Up | End-to-end testing, docs, git | Day 7 |

---

---

## 🟦 Phase 1 — Project Setup & Data Ingestion

### Goal
Establish the project structure, initialize git, and prepare the Zomato dataset in a usable local format.

### Tasks

#### 1.1 Initialize Project Structure
Create the following directory and file layout:
```text
02. Zomato project/
├── index.html
├── context.md           ✅ Done
├── architecture.md      ✅ Done
├── problemStatement.txt ✅ Done
├── implementation_plan.md ✅ Done
├── css/
│   └── styles.css
├── js/
│   ├── app.js
│   ├── filters.js
│   ├── recommendations.js
│   └── config.js        ← gitignored
├── data/
│   └── zomato_dataset.json
└── assets/
    └── icons/
```

#### 1.2 Initialize Git Repository
```bash
git init
echo "js/config.js" >> .gitignore
echo "*.DS_Store" >> .gitignore
git add .
git commit -m "chore: initial project scaffold"
```

#### 1.3 Download & Preprocess Zomato Dataset
- Source: [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- Download via HuggingFace Datasets or the dataset's "Files" tab
- Preprocess and save as `data/zomato_dataset.json`

**Required fields to retain per record:**
```json
{
  "name": "The Spice Route",
  "location": "Delhi",
  "cuisines": "North Indian, Mughlai",
  "cost_for_two": 600,
  "aggregate_rating": 4.5,
  "votes": 1200,
  "highlights": "Family-friendly, Live Music"
}
```

**Preprocessing steps:**
- Drop records with missing `name`, `location`, or `aggregate_rating`
- Normalize `cost_for_two` to numeric integer (strip `₹`, commas)
- Trim whitespace from all string fields
- Standardize cuisine labels to Title Case

### ✅ Phase 1 Done When
- [ ] Directory structure exists
- [ ] Git initialized with `.gitignore`
- [ ] `zomato_dataset.json` saved with clean, normalized records

---

---

## 🟩 Phase 2 — UI & Input Layer

### Goal
Build the full HTML skeleton and CSS design system. Implement the user preference form with all input fields.

### Tasks

#### 2.1 Design Tokens in `css/styles.css`
```css
:root {
  --color-brand:        #E23744;   /* Zomato Red */
  --color-brand-dark:   #C0212E;
  --color-bg:           #0F0F0F;   /* Dark background */
  --color-surface:      #1A1A1A;
  --color-surface-2:    #242424;
  --color-text:         #F5F5F5;
  --color-text-muted:   #9CA3AF;
  --color-border:       #2E2E2E;
  --color-success:      #22C55E;
  --font-primary:       'Inter', sans-serif;
  --radius-card:        12px;
  --shadow-card:        0 4px 24px rgba(0,0,0,0.4);
  --transition:         0.2s ease;
}
```

#### 2.2 HTML Structure (`index.html`)
Build the following sections:

| Section | Description |
|---|---|
| `<nav>` | Logo + app name (`ZomaAI`) |
| `<section class="hero">` | Tagline, subtitle |
| `<section class="form-section">` | User preference input form |
| `<section class="results-section">` | Empty initially — populated by JS |
| `<footer>` | Credits, dataset source link |

#### 2.3 Input Form Fields

| Field | Input Type | Values |
|---|---|---|
| Location | `<input type="text">` | Free text (e.g., Delhi) |
| Budget | `<select>` | Low / Medium / High |
| Cuisine | `<input type="text">` | Free text (e.g., Italian) |
| Min Rating | `<input type="range" min="1" max="5">` | 1.0 – 5.0 with live label |
| Additional Preferences | `<input type="text">` | Free text (e.g., family-friendly) |
| Submit Button | `<button id="find-btn">` | "Find Restaurants 🚀" |

#### 2.4 Responsive Layout
- Mobile-first CSS grid
- Form stacks vertically on `< 600px`
- Results grid: 1 column mobile, 2 columns tablet, 3 columns desktop

### ✅ Phase 2 Done When
- [ ] All form fields render correctly
- [ ] Design tokens applied — dark mode active
- [ ] Page is fully responsive across breakpoints
- [ ] Google Font (`Inter`) loaded

---

---

## 🟨 Phase 3 — Filtering Engine

### Goal
Implement `filters.js` to narrow down the full Zomato dataset to the top N most relevant candidate restaurants based on user input.

### Tasks

#### 3.1 Load Dataset
```javascript
// filters.js
let dataset = [];

export async function loadDataset() {
  const res = await fetch('./data/zomato_dataset.json');
  dataset = await res.json();
}
```

#### 3.2 Budget Band Mapping
```javascript
const BUDGET_MAP = {
  low:    { min: 0,   max: 300  },
  medium: { min: 300, max: 700  },
  high:   { min: 700, max: Infinity }
};
```

#### 3.3 Filter Pipeline
```
filterRestaurants(userPrefs) {
  Step 1: location  → record.location includes userPrefs.location (case-insensitive)
  Step 2: budget    → record.cost_for_two within BUDGET_MAP[userPrefs.budget] range
  Step 3: cuisine   → record.cuisines includes userPrefs.cuisine (case-insensitive)
  Step 4: rating    → record.aggregate_rating >= userPrefs.minRating
  Step 5: sort DESC by votes
  Step 6: return top 12 results
}
```

#### 3.4 Edge Cases
- If location yields 0 results → relax location filter, try nationwide
- If cuisine yields 0 results → skip cuisine filter, keep others
- Minimum 5 results passed to LLM; if < 5, relax filters progressively

### ✅ Phase 3 Done When
- [ ] `loadDataset()` successfully fetches and parses JSON
- [ ] All 4 filters work individually and in combination
- [ ] Edge cases handled — no crashes on zero results
- [ ] Returns top 12 candidate records to caller

---

---

## 🟧 Phase 4 — Groq LLM Integration

### Goal
Build the prompt construction logic and Groq API integration in `recommendations.js`. Parse the LLM's JSON response for rendering.

### Tasks

#### 4.1 Groq API Configuration (`config.js`)
```javascript
// js/config.js  ← DO NOT COMMIT
export const GROQ_API_KEY = "gsk_your_key_here";
export const GROQ_MODEL   = "llama3-8b-8192";  // or "mixtral-8x7b-32768"
export const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";
```

#### 4.2 Prompt Builder
```javascript
function buildPrompt(userPrefs, candidates) {
  const candidateText = candidates.map((r, i) =>
    `${i+1}. ${r.name} | ${r.location} | ${r.cuisines} | ` +
    `Rating: ${r.aggregate_rating} | Cost for 2: ₹${r.cost_for_two} | ` +
    `Highlights: ${r.highlights || 'N/A'}`
  ).join('\n');

  return `
You are a restaurant recommendation assistant.
A user is looking for a restaurant with:
  - Location: ${userPrefs.location}
  - Budget: ${userPrefs.budget}
  - Cuisine: ${userPrefs.cuisine}
  - Minimum Rating: ${userPrefs.minRating}
  - Additional Preferences: ${userPrefs.extras || 'None'}

Here are ${candidates.length} candidate restaurants:
${candidateText}

Task:
1. Rank the top 3–5 most suitable restaurants for this user.
2. For each, provide a 2–3 sentence explanation of why it fits.
3. Respond ONLY in valid JSON using this schema:
[
  {
    "rank": 1,
    "name": "...",
    "cuisine": "...",
    "rating": 4.5,
    "cost_for_two": "₹600",
    "explanation": "..."
  }
]
  `.trim();
}
```

#### 4.3 Groq API Call
```javascript
async function callGroq(prompt) {
  const response = await fetch(GROQ_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
      max_tokens: 1024
    })
  });
  const data = await response.json();
  return data.choices[0].message.content;
}
```

#### 4.4 Response Parsing
```javascript
function parseRecommendations(raw) {
  try {
    // Strip markdown code fences if present
    const clean = raw.replace(/```json|```/g, '').trim();
    return JSON.parse(clean);
  } catch (e) {
    console.error("Failed to parse LLM response:", e);
    return null;
  }
}
```

### ✅ Phase 4 Done When
- [ ] `config.js` contains valid Groq API key (tested manually)
- [ ] Prompt builder produces correct formatted string
- [ ] Groq API returns a response successfully (console.log verified)
- [ ] `parseRecommendations()` correctly parses JSON from LLM output
- [ ] Handles malformed / non-JSON responses gracefully

---

---

## 🟥 Phase 5 — Output Rendering

### Goal
Render the parsed LLM recommendations as styled, animated cards in the results section.

### Tasks

#### 5.1 Card HTML Template (injected by JS)
```html
<div class="rec-card" id="rec-card-{rank}">
  <div class="rec-rank">🥇 #{rank}</div>
  <h2 class="rec-name">{name}</h2>
  <div class="rec-meta">
    <span class="rec-cuisine">🥘 {cuisine}</span>
    <span class="rec-rating">⭐ {rating}</span>
    <span class="rec-cost">💰 {cost_for_two}</span>
  </div>
  <p class="rec-explanation">🤖 {explanation}</p>
</div>
```

#### 5.2 Loading State
- Show a spinner / skeleton loader while waiting for Groq API response
- Disable the submit button during loading

#### 5.3 Animation
- Cards fade-in with staggered delay using CSS `@keyframes`
- Each card slides up subtly on entry (`translateY(20px) → 0`)

#### 5.4 Empty / Error State
```
"No matching restaurants found. Try broadening your preferences."
"Could not reach the AI service. Please try again shortly."
```

### ✅ Phase 5 Done When
- [ ] Recommendation cards render with all required fields
- [ ] Loading spinner shown during API call
- [ ] Staggered card entrance animations working
- [ ] Empty state and error state messages display correctly

---

---

## 🟪 Phase 6 — Polish & Error Handling

### Goal
Elevate the UX with micro-animations, dark mode refinements, persistence, and robust error handling.

### Tasks

#### 6.1 UX Enhancements
- [ ] Persist last search in `localStorage` — auto-fill on reload
- [ ] Rating slider shows live value (e.g., "⭐ 4.0+")
- [ ] "Clear Results" button resets view
- [ ] Smooth scroll to results on submit

#### 6.2 Micro-Animations
- [ ] Submit button pulse effect on hover
- [ ] Form field focus glow (brand red outline)
- [ ] Card hover: subtle lift (`box-shadow` + `translateY(-4px)`)
- [ ] Skeleton loader shimmer while fetching

#### 6.3 Error Handling
- [ ] Network timeout → show retry button
- [ ] Groq API error (non-200) → user-friendly message with status code
- [ ] JSON parse failure → fallback to showing raw text in a monospace block
- [ ] Dataset load failure → alert with dataset link to manual download

#### 6.4 Accessibility
- [ ] All form inputs have `<label>` associations
- [ ] Focus states visible for keyboard navigation
- [ ] ARIA live region on `results-section` for screen reader updates

### ✅ Phase 6 Done When
- [ ] Last search persists across page reloads
- [ ] All animations implemented and smooth
- [ ] All error cases produce user-friendly feedback
- [ ] Basic accessibility criteria met

---

---

## ⬛ Phase 7 — Testing & Wrap-Up

### Goal
End-to-end testing of all user flows, documentation cleanup, and final git commit.

### Tasks

#### 7.1 Manual Test Cases

| Test Case | Input | Expected Output |
|---|---|---|
| Valid full input | Delhi, Medium, North Indian, 4.0 | Top 3–5 ranked cards with explanations |
| Unknown location | XYZ City | Relaxed filter, nationwide results or graceful empty state |
| High budget, rare cuisine | High, Ethiopian | Handles sparse results, shows fallback if needed |
| Network offline | Any | Error message + retry button |
| Low rating threshold | Min 1.0 | Many candidates, LLM picks best |
| Missing optional field | No extra preferences | Works without extras field |

#### 7.2 Cross-Browser Check
- [ ] Chrome ✓
- [ ] Firefox ✓
- [ ] Safari ✓
- [ ] Mobile (Chrome on Android / Safari on iOS) ✓

#### 7.3 Final Documentation
- [ ] Update `context.md` with final tech decisions
- [ ] Update `architecture.md` with any changed components
- [ ] Add `README.md` with setup instructions and Groq API key steps

#### 7.4 Git Final Commit
```bash
git add .
git commit -m "feat: complete AI restaurant recommendation system v1.0"
git remote add origin <your-repo-url>
git push -u origin main
```

### ✅ Phase 7 Done When
- [ ] All manual test cases pass
- [ ] Cross-browser verified
- [ ] README written
- [ ] Final commit pushed to GitHub

---

---

## 📊 Summary Checklist

| Phase | Status |
|---|---|
| Phase 1 — Setup & Data Ingestion | ⬜ Not Started |
| Phase 2 — UI & Input Layer | ⬜ Not Started |
| Phase 3 — Filtering Engine | ⬜ Not Started |
| Phase 4 — Groq LLM Integration | ⬜ Not Started |
| Phase 5 — Output Rendering | ⬜ Not Started |
| Phase 6 — Polish & Error Handling | ⬜ Not Started |
| Phase 7 — Testing & Wrap-Up | ⬜ Not Started |

---

> **Tip:** Work through phases sequentially. Each phase produces a testable deliverable before moving to the next.
