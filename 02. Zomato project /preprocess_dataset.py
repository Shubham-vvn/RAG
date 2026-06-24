"""
preprocess_dataset.py — Zomato Dataset Downloader & Preprocessor
==================================================================

Phase 1, Task 1.3 — Downloads the Zomato restaurant dataset from
HuggingFace and preprocesses it into a clean JSON file for use
by the frontend filtering engine (filters.js).

Usage:
  pip install datasets pandas
  python preprocess_dataset.py

Output:
  data/zomato_dataset.json

Dataset Source:
  https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation
"""

import json
import os
import re

# ── Try importing required libraries ────────────────────────────────────────
try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("❌ Missing dependencies. Run: pip install datasets pandas")
    raise SystemExit(1)

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_ID  = "ManikaSaini/zomato-restaurant-recommendation"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zomato_dataset.json")

# Fields to retain in the output JSON
REQUIRED_FIELDS = ["name", "location", "cuisines", "cost_for_two",
                   "aggregate_rating", "votes", "highlights"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_cost(value) -> int:
    """
    Normalize cost_for_two to a plain integer.
    Handles: '₹600', '1,200', 600, None
    """
    if value is None:
        return 0
    s = str(value).replace("₹", "").replace(",", "").strip()
    try:
        return int(float(s))
    except ValueError:
        return 0


def normalize_cuisines(value) -> str:
    """
    Normalize cuisine separators and apply Title Case.
    Handles: 'north indian/mughlai', 'italian | chinese', etc.
    """
    if not value or str(value).strip().lower() in ("nan", "none", ""):
        return "Various"
    # Replace alternative separators with comma
    cleaned = re.sub(r"\s*[/|]\s*", ", ", str(value))
    # Title case each cuisine
    return ", ".join(c.strip().title() for c in cleaned.split(",") if c.strip())


def normalize_rating(value) -> float:
    """
    Parse aggregate_rating to float. Returns 0.0 on failure.
    """
    try:
        rating = float(value)
        return round(max(0.0, min(5.0, rating)), 1)  # Clamp to [0, 5]
    except (TypeError, ValueError):
        return 0.0


def normalize_votes(value) -> int:
    """
    Parse votes to int. Returns 0 on failure.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def deduplicate(records: list) -> list:
    """
    Remove duplicate entries by composite key: (name + location).
    """
    seen = set()
    unique = []
    for r in records:
        key = (r["name"].lower().strip(), r["location"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ── Main Preprocessing Pipeline ──────────────────────────────────────────────

def preprocess():
    print(f"📥 Downloading dataset: {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID, split="train")
    df  = raw.to_pandas()

    print(f"✅ Downloaded {len(df)} raw records")
    print(f"   Columns available: {list(df.columns)}")

    # ── Step 1: Drop records missing critical fields ─────────────────────────
    required_cols = ["name", "location", "aggregate_rating"]
    before = len(df)
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    print(f"   After dropping nulls: {len(df)} records (removed {before - len(df)})")

    # ── Step 2: Normalize all fields ─────────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        record = {
            "name":             str(row.get("name", "")).strip(),
            "location":         str(row.get("location", "")).strip(),
            "cuisines":         normalize_cuisines(row.get("cuisines")),
            "cost_for_two":     normalize_cost(row.get("cost_for_two", row.get("approx_cost(for two people)", 0))),
            "aggregate_rating": normalize_rating(row.get("aggregate_rating", row.get("rate", 0))),
            "votes":            normalize_votes(row.get("votes", 0)),
            "highlights":       str(row.get("highlights", row.get("dish_liked", ""))).strip(),
        }

        # Skip records with blank name or location after normalization
        if not record["name"] or record["name"].lower() in ("nan", "none"):
            continue
        if not record["location"] or record["location"].lower() in ("nan", "none"):
            continue

        records.append(record)

    print(f"   After normalization: {len(records)} valid records")

    # ── Step 3: Deduplicate ──────────────────────────────────────────────────
    records = deduplicate(records)
    print(f"   After deduplication: {len(records)} unique records")

    # ── Step 4: Save output ──────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(records)} records → {OUTPUT_FILE}")
    print("\nSample record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    preprocess()
