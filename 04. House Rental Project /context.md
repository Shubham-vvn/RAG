# Context: AI-Powered Rental Property Recommendation System

## Overview
You are building an AI-powered rental property recommendation service tailored for the Indian housing market. By combining structured filtering on database fields with the reasoning capabilities of a Large Language Model (LLM), the system will intelligently suggest properties based on both hard constraints (budget, size, location) and soft user preferences (vibe, amenities, specific needs).

---

## Objectives
- **Targeted Filtering:** Programmatically filter property listings based on explicit constraints (e.g., city, budget range, BHK configuration, furnishing status, tenant type).
- **LLM Reasoning & Personalization:** Leverage an LLM to evaluate soft preferences (e.g., "close to tech parks," "quiet neighborhood," "spacious balcony") against property details.
- **Explainable Recommendations:** Rank the shortlisted properties and generate human-like explanations of why they match the user's needs, including highlighting any trade-offs.
- **Modern User Interface:** Present the filtered results and personalized explanations in a clean, intuitive layout.

---

## System Architecture & Workflow

### 1. Data Ingestion & Preprocessing
* **Source Dataset:** [House_Rent_Dataset.csv](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project/House_Rent_Dataset.csv) (containing ~4,700+ structured listings).
* **Target Fields:** `BHK`, `Rent`, `Size`, `Floor`, `Area Type`, `Area Locality`, `City`, `Furnishing Status`, `Tenant Preferred`, `Bathroom`, and `Point of Contact`.
* **Data Cleaning:** 
  * Parse complex columns (e.g., splitting "Floor" strings like `"2 out of 5"` into numeric `floor` and `total_floors` fields).
  * Deduplicate listings and handle missing values.
* **Offline Synthetic Enrichment:** 
  * Because the original dataset is purely structured, generate a synthetic layer offline *once* using an LLM.
  * For each listing, generate a simulated **short description**, **tenant review**, and **inferred list of amenities**.
  * Cache these enriched listings in a new CSV/database to avoid repeating expensive API calls during user queries.

### 2. User Input Collection
Gather key user criteria across:
* **Hard Filters:** City, Budget Range (min/max monthly rent), BHK Configuration (1, 2, 3, 4+ BHK), Furnishing Status (Furnished, Semi-Furnished, Unfurnished), and Tenant Type (Bachelors, Family, Company).
* **Soft Preferences:** Natural language input describing specific vibes or features (e.g., "near public transit," "well-ventilated," "good schools nearby").

### 3. Integration Layer
* **Phase 1: Hard Filtering (Pandas/SQL):** Filter the dataset programmatically. Do NOT use the LLM for hard constraints, as programmatic filters are 100% reliable and fast.
* **Phase 2: Shortlisting:** Select a top-N subset (e.g., 5–10 matching candidates) from the filtered results.
* **Phase 3: Prompt Construction:** Package the shortlist, along with their synthetic descriptions, reviews, and amenities, into a structured LLM prompt alongside the user's soft preferences.

### 4. Recommendation Engine (LLM)
* Rank the short-listed properties based on overall fit.
* Generate explanations demonstrating why each property is recommended.
* Identify and present trade-offs (e.g., *"Property A is closer to your office, but Property B offers more space for a lower rent"*).

### 5. Output Display
Present recommendations showing:
* Locality & City
* BHK / Size / Furnishing Status
* Monthly Rent
* Personalized AI-generated explanation of the fit

---

## Dataset Constraints & Notes
* **File Location:** `House_Rent_Dataset.csv` is locally located in the root workspace.
* **Cache Requirement:** Synthetic descriptions, reviews, and amenities *must* be pre-generated offline and cached.
* **License:** Released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (attribution to the original Kaggle author is required).
* **Portability:** For distribution, relocate the CSV to a `./data/` folder and update files to use relative paths.
