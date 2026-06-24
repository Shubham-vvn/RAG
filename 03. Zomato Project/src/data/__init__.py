"""
Data ingestion layer — dataset loading, preprocessing, and repository.

Usage:
    from src.data import initialize_data

    repository = initialize_data()
    print(f"Loaded {repository.get_count()} restaurants")
"""

import logging

logger = logging.getLogger(__name__)


def initialize_data():
    """
    Load, preprocess, and cache the dataset. Returns a ready-to-query repository.

    Pipeline: DatasetLoader → DataPreprocessor → RestaurantRepository

    Returns:
        RestaurantRepository: In-memory query interface over preprocessed data.
    """
    from src.data.loader import DatasetLoader
    from src.data.preprocessor import DataPreprocessor
    from src.data.repository import RestaurantRepository

    # Step 1: Load raw data (from cache or Hugging Face)
    loader = DatasetLoader()
    raw_df = loader.load()
    logger.info(f"Raw dataset: {len(raw_df)} rows")

    # Step 2: Preprocess to canonical schema
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)
    logger.info(f"Preprocessed dataset: {len(clean_df)} rows")

    # Step 3: Build in-memory repository
    repository = RestaurantRepository.from_dataframe(clean_df)
    logger.info(
        f"Repository initialized: {repository.get_count()} restaurants, "
        f"{len(repository.get_locations())} locations, "
        f"{len(repository.get_cuisines())} cuisines"
    )

    return repository


__all__ = ["initialize_data"]
