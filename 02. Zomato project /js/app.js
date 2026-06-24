/**
 * app.js — ZomaAI Main Coordinator
 *
 * Responsibilities:
 *  - Initialize the application
 *  - Listen for form submit events
 *  - Orchestrate: Input → Filter → Prompt → LLM → Render
 *
 * Phase 2 will wire up event listeners.
 * Phase 4 will wire up full LLM call pipeline.
 */

import { loadDataset, filterRestaurants } from './filters.js';
import { getRecommendations, renderCards } from './recommendations.js';

// ── App Entry Point ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  console.log('[ZomaAI] App initialized');
  await loadDataset();
  console.log('[ZomaAI] Dataset loaded successfully');

  // Event listener wired up in Phase 2
});
