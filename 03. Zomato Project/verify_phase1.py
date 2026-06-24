import logging
import sys
from src.data import initialize_data

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    print("=== First Run: Downloading and Preprocessing ===")
    repo = initialize_data()
    print(f"Loaded {repo.get_count()} restaurants successfully!")
    print(f"Distinct Locations ({len(repo.get_locations())}): {repo.get_locations()[:10]}")
    print(f"Distinct Cuisines ({len(repo.get_cuisines())}): {repo.get_cuisines()[:10]}")

    print("\n=== Second Run: Loading from Parquet Cache ===")
    repo2 = initialize_data()
    print(f"Loaded {repo2.get_count()} restaurants from cache successfully!")
    assert repo.get_count() == repo2.get_count(), "Restaurant counts do not match!"

if __name__ == "__main__":
    main()
