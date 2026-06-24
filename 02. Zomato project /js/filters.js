/**
 * filters.js — Dataset Filtering Engine
 *
 * Responsibilities:
 *  - Load and cache zomato_dataset.json
 *  - Apply user preference filters (location, budget, cuisine, rating)
 *  - Return top N candidate restaurants for the LLM
 *
 * Full implementation in Phase 3.
 */

// ── Budget Band Mapping ──────────────────────────────────────
export const BUDGET_MAP = {
  low:    { min: 0,   max: 300 },
  medium: { min: 300, max: 700 },
  high:   { min: 700, max: Infinity },
};

// ── In-memory dataset cache ──────────────────────────────────
let dataset = [];

/**
 * loadDataset()
 * Fetches and caches the Zomato JSON dataset.
 * Edge case: Handles missing file (404) and malformed JSON.
 */
export async function loadDataset() {
  try {
    const res = await fetch('./data/zomato_dataset.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}: dataset not found`);
    dataset = await res.json();
    if (!Array.isArray(dataset) || dataset.length === 0) {
      throw new Error('Dataset is empty or not a valid array');
    }
    console.log(`[filters] Loaded ${dataset.length} restaurant records`);
  } catch (err) {
    console.error('[filters] Failed to load dataset:', err.message);
    dataset = []; // Graceful degradation
  }
}

/**
 * filterRestaurants(userPrefs)
 * Applies a 4-step filter pipeline and returns top N candidates.
 *
 * @param {Object} userPrefs - { location, budget, cuisine, minRating, extras }
 * @param {number} topN      - Max candidates to return (default: 12)
 * @returns {Array}          - Filtered & sorted restaurant records
 */
export function filterRestaurants(userPrefs, topN = 12) {
  // Phase 3 will implement the full pipeline
  console.log('[filters] filterRestaurants() — implementation in Phase 3');
  return [];
}
