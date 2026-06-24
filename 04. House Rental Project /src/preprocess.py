import os
import re
import pandas as pd

def parse_floor_string(floor_str):
    """
    Parses a floor string like 'Ground out of 2' or '3 out of 5' or 'Ground'
    into a tuple of (floor_num, total_floors).
    """
    if pd.isna(floor_str):
        return 0, 1  # Default fallback: Ground floor, 1 story
    
    floor_str = str(floor_str).strip().lower()
    
    # Check for "X out of Y" format
    match = re.match(r"([\w\s]+)\s+out\s+of\s+(\d+)", floor_str)
    if match:
        floor_raw, total_raw = match.groups()
        try:
            total_floors = int(total_raw)
        except ValueError:
            total_floors = 1
            
        if "ground" in floor_raw:
            floor_num = 0
        elif "lower basement" in floor_raw:
            floor_num = -2
        elif "upper basement" in floor_raw or "basement" in floor_raw:
            floor_num = -1
        else:
            try:
                floor_num = int(floor_raw)
            except ValueError:
                floor_num = 1  # Safe default
                
        # Enforce consistency: floor_num should not exceed total_floors
        if floor_num > total_floors:
            total_floors = floor_num
            
        return floor_num, total_floors
    
    # If it is just a plain digit (e.g. "3")
    if floor_str.isdigit():
        val = int(floor_str)
        return val, val
    
    # Handle single string descriptions
    if "lower basement" in floor_str:
        return -2, 1
    if "upper basement" in floor_str or "basement" in floor_str:
        return -1, 1
    if "ground" in floor_str:
        return 0, 1
        
    return 1, 1  # Safe default fallback

def preprocess_dataset(input_path, output_path):
    """
    Loads, cleans, deduplicates, and parses the dataset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    initial_rows = len(df)
    
    # 1. Deduplicate rows
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"Removed {initial_rows - dedup_rows} duplicate rows.")
    
    # 2. Drop rows with nulls in critical columns
    critical_cols = ['BHK', 'Rent', 'Size', 'City', 'Floor']
    df = df.dropna(subset=critical_cols)
    print(f"Dropped rows with null values in critical columns. Remaining: {len(df)}")
    
    # 3. Clean numeric fields and handle sanity bounds
    df['BHK'] = df['BHK'].astype(int)
    df['Rent'] = df['Rent'].astype(int)
    df['Size'] = df['Size'].astype(int)
    df['Bathroom'] = df['Bathroom'].fillna(1).astype(int) # Default 1 bathroom if null
    
    # Enforce sanity limits
    df = df[(df['Rent'] > 0) & (df['Size'] > 0) & (df['BHK'] > 0)]
    print(f"Applied numeric sanity bounds (Rent, Size, BHK > 0). Remaining: {len(df)}")
    
    # 4. Parse Floor column
    parsed_floors = df['Floor'].apply(parse_floor_string)
    df['floor_num'] = [x[0] for x in parsed_floors]
    df['total_floors'] = [x[1] for x in parsed_floors]
    
    # 5. Fill missing values for non-critical categorical columns
    df['Furnishing Status'] = df['Furnishing Status'].fillna('Unfurnished').str.strip()
    df['Tenant Preferred'] = df['Tenant Preferred'].fillna('Bachelors/Family').str.strip()
    df['Area Locality'] = df['Area Locality'].fillna('Unknown Locality').str.strip()
    df['Point of Contact'] = df['Point of Contact'].fillna('Contact Owner').str.strip()
    
    # Add an ID column if it doesn't exist
    df = df.reset_index(drop=True)
    df['Listing_Id'] = df.index + 1
    
    # Reorder columns to put Listing_Id first
    cols = ['Listing_Id'] + [col for col in df.columns if col != 'Listing_Id']
    df = df[cols]
    
    print(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Preprocessing completed successfully!")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "../data/House_Rent_Dataset.csv")
    output_file = os.path.join(current_dir, "../data/house_rent_clean.csv")
    preprocess_dataset(input_file, output_file)
