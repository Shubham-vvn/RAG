# Implementation Plan: AI-Powered Rental Property Recommendation System

This implementation plan outlines the steps to build the property recommendation service using Python, Pandas, Streamlit, and Groq API.

## User Review Required

> [!IMPORTANT]
> The raw dataset contains over 4,700 rows. Running Groq LLM calls to enrich all 4,700 rows with synthetic descriptions, reviews, and amenities will take substantial time and may hit API rate limits.
> We propose generating the enrichment layer for a **representative subset (e.g., top 100-200 listings)** or implementing a robust batching and checkpointing script to run it incrementally. Please confirm if you would like to enrich the entire dataset or a smaller subset first.

---

## Proposed Changes

### 1. Project Directory Structure
We will organize the repository to ensure portability and modularity.

#### [NEW] [data/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/data/)
* Store the relocated dataset and cached enriched outputs.

#### [NEW] [src/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/src/)
* Contain the backend logic, offline parsing/enrichment scripts, and testing suites.

---

### Phase 1: Environment Setup & Data Portability
- **Tasks:**
  - Create standard directories: `./data` and `./src`.
  - Move the raw `House_Rent_Dataset.csv` into `./data/House_Rent_Dataset.csv`.
  - Initialize python virtual environment and `requirements.txt` containing:
    - `groq`
    - `pandas`
    - `streamlit`
    - `python-dotenv`
    - `pytest`

---

### Phase 2: Offline Preprocessing & Enrichment (Groq)
- **Tasks:**
  - Write `src/preprocess.py` to clean the dataset:
    - Parse `"Floor"` column (e.g., `"2 out of 5"` -> `floor_num: 2`, `total_floors: 5`).
    - Handle null/missing values and duplicates.
  - Write `src/enrich.py` to call Groq API offline:
    - Set up a prompt requesting: Short Description, Tenant Review, and Amenities List.
    - Integrate a batch-processing loop with **checkpointing** (saving progress to a temp file periodically so it can resume if interrupted).
    - Save output to `./data/house_rent_enriched.csv`.

---

### Phase 3: Programmatic Filtering & Prompt Engineering
- **Tasks:**
  - Write `src/filter.py` to implement hard filters using Pandas:
    - Inputs: city, budget min/max, BHK, furnishing status, tenant preference.
    - Output: Programmatically filtered subset dataframe.
  - Write `src/recommender.py` to handle Groq API integration for recommendations:
    - Shortlist top N candidates (e.g., top 5-10 properties matching budget/BHK).
    - Construct the LLM recommendation prompt injecting listing details (clean fields + enriched text) and user soft preferences.
    - Define response parser to capture ranked properties, match descriptions, and trade-offs.

---

### Phase 4: Streamlit Frontend UI
- **Tasks:**
  - Create `app.py` at the root directory:
    - **Sidebar Input Panel:**
      - Dropdown for City.
      - Slider for Rent Range (Budget).
      - Select boxes for BHK, Furnishing Status, Tenant Preferred.
      - Free text field for Soft Preferences (vibe, proximity, amenities).
    - **Main Results Panel:**
      - Display loading animations while processing.
      - Render property recommendation cards showing key metrics (Rent, BHK, Location, Size).
      - Display the Groq-generated personalized explanation and trade-offs.

---

## Verification Plan

### Automated Tests
- Write unit tests in `tests/test_preprocess.py` and `tests/test_filter.py` to verify:
  - `"Floor"` string parsing helper functions.
  - Pandas hard filtering accuracy.

### Manual Verification
- Run the offline enrichment script on a small batch of 10 rows and verify the CSV cache format.
- Launch the Streamlit application using:
  ```bash
  streamlit run app.py
  ```
- Test query variations (e.g., low budget, high budget, family vs bachelor preferences) to verify Groq's ranking logic and text formatting.
