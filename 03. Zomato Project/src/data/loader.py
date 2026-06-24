"""
Hugging Face dataset loader with local parquet caching.

Loads the Zomato restaurant dataset from Hugging Face on first run,
caches it locally as parquet for fast subsequent loads.

Implementation: Phase 1 (Data Ingestion Layer)
"""

import logging
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import settings

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads the Zomato dataset from Hugging Face or local cache."""

    def __init__(self, cache_path: str = None):
        self.cache_path = Path(cache_path or settings.DATA_CACHE_PATH)

    def load(self) -> pd.DataFrame:
        """Load dataset from cache or Hugging Face."""
        if self.cache_path.exists():
            logger.info(f"Loading dataset from cache: {self.cache_path}")
            start = time.time()
            df = pd.read_parquet(self.cache_path)
            logger.info(f"Loaded {len(df)} rows from cache in {time.time() - start:.2f}s")
            return df

        logger.info(f"Cache not found. Downloading from Hugging Face: {settings.HF_DATASET_NAME}")
        return self._download_and_cache()

    def _download_and_cache(self) -> pd.DataFrame:
        """Download from Hugging Face with retry, then cache locally."""
        max_retries = settings.GROQ_MAX_RETRIES  # Reuse general max retries config
        for attempt in range(1, max_retries + 1):
            try:
                start = time.time()
                dataset = load_dataset(settings.HF_DATASET_NAME, split=settings.HF_DATASET_SPLIT)
                df = dataset.to_pandas()
                elapsed = time.time() - start
                logger.info(f"Downloaded {len(df)} rows in {elapsed:.2f}s (attempt {attempt})")

                # Cache locally
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(self.cache_path, index=False)
                logger.info(f"Cached dataset to {self.cache_path}")
                return df

            except Exception as e:
                logger.warning(f"Download attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download dataset after {max_retries} attempts") from e
                time.sleep(2 ** attempt)  # Exponential backoff

    def clear_cache(self):
        """Delete local cache to force re-download."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info(f"Cache cleared: {self.cache_path}")

