import os
import sys
import json
import time
import argparse
import pandas as pd

# Try to load dotenv safely
try:
    from dotenv import load_dotenv
    load_dotenv()
    has_dotenv = True
except ImportError:
    has_dotenv = False

# Try to load groq safely
try:
    from groq import Groq, GroqError
    has_groq = True
except ImportError:
    has_groq = False
    class GroqError(Exception):
        pass

# Check for API key
api_key = os.getenv("GROQ_API_KEY")
is_mock_mode = False

# Determine if we should run in mock mode
if not has_groq or not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
    is_mock_mode = True
    if not has_groq:
        print("--- [WARNING] groq library is not installed. Falling back to local MOCK ENRICHMENT MODE ---")
    else:
        print("--- [WARNING] GROQ_API_KEY is not set or is using placeholder. Falling back to local MOCK ENRICHMENT MODE ---")

def generate_mock_enrichment(row):
    """
    Generates realistic structured mock enrichment data locally based on property characteristics.
    """
    bhk = row['BHK']
    rent = row['Rent']
    city = row['City']
    locality = row['Area Locality']
    furnishing = str(row['Furnishing Status']).strip().lower()
    
    # 1. Amenities based on rent and furnishing
    amenities = ["24/7 Water Supply", "Electricity Backup"]
    if furnishing == "furnished":
        amenities += ["Sofa Set", "Double Bed", "Modular Kitchen", "Air Conditioner", "Geyser"]
    elif furnishing == "semi-furnished":
        amenities += ["Wardrobes", "Geyser", "Ceiling Fans & Lights"]
    else:
        amenities += ["Exhaust Fan"]
        
    if rent >= 50000:
        amenities += ["Gymnasium", "Swimming Pool", "24/7 Multi-tier Security", "Elevator", "Club House", "Power Backup"]
    elif rent >= 25000:
        amenities += ["Covered Car Parking", "CCTV Security", "Intercom", "WiFi Connectivity"]
    else:
        amenities += ["Open Parking", "Balcony"]
        
    # Deduplicate amenities
    amenities = list(sorted(set(amenities)))
    
    # 2. Descriptions based on parameters
    descriptions = [
        f"A beautiful {bhk} BHK {furnishing} apartment in the sought-after area of {locality}, {city}. Perfect for those seeking comfortable living.",
        f"This spacious {bhk} BHK property offers comfortable living with close proximity to essential utilities, markets, and transit in {locality}.",
        f"An affordable and well-maintained {bhk} BHK home located in {locality}, ideal for families or working professionals looking for a cozy space."
    ]
    # Pick a description deterministically based on Listing_Id
    desc = descriptions[int(row['Listing_Id']) % len(descriptions)]
    
    # 3. Reviews
    reviews = [
        f"Lived in this {locality} property for a year. Excellent community, very peaceful locality, and all amenities are easily accessible.",
        f"The house is very spacious and well-ventilated. The landlord is extremely cooperative, though parking can sometimes be a challenge.",
        f"Great location close to public transit and local markets. Power backup is reliable and security is good."
    ]
    rev = reviews[int(row['Listing_Id']) % len(reviews)]
    
    return {
        "description": desc,
        "review": rev,
        "amenities": amenities
    }

def call_groq_enrichment(client, row, model="llama-3.1-8b-instant"):
    """
    Calls Groq API to generate synthetic description, review, and amenities.
    Falls back to mock mode if authentication fails.
    """
    global is_mock_mode
    if is_mock_mode:
        return generate_mock_enrichment(row)
        
    prompt = f"""
    You are an expert real estate consultant. Generate synthetic enriched text data for a rental listing in India based on the following structured details:
    - City: {row['City']}
    - Locality: {row['Area Locality']}
    - BHK: {row['BHK']} BHK
    - Monthly Rent: INR {row['Rent']}
    - Size: {row['Size']} Sq. Ft.
    - Furnishing Status: {row['Furnishing Status']}
    - Tenant Preferred: {row['Tenant Preferred']}
    - Bathrooms: {row['Bathroom']}
    
    You must output a raw JSON object with exactly three fields:
    1. "description": A short, catchy, realistic marketing description for this rental (1 to 2 sentences).
    2. "review": A simulated tenant review from someone who previously lived there (2 sentences, detailing positive/negative vibes).
    3. "amenities": A list of 4 to 8 specific, realistic amenities inferred for this property (e.g., ["24/7 Water", "Geyser", "Covered Parking", "Modular Kitchen"]). Ensure amenities match the budget (INR {row['Rent']}) and location.
    
    Do not write any markdown code blocks or explanations outside the JSON. Return only the JSON object.
    """
    
    retries = 3
    backoff = 2
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs only structured JSON data. Do not include markdown code block formatting like ```json ... ```."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except GroqError as e:
            # Check for API Key or auth issue (401)
            if "401" in str(e) or "invalid api key" in str(e).lower():
                print("--- [ERROR] Invalid Groq API Key (401). Switching to MOCK MODE for remainder of execution. ---")
                is_mock_mode = True
                return generate_mock_enrichment(row)
                
            if "429" in str(e) or "rate limit" in str(e).lower():
                sleep_time = backoff * (2 ** attempt)
                print(f"Rate limited. Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"Groq API Error: {e}. Retrying...")
                time.sleep(2)
        except json.JSONDecodeError:
            print(f"JSON decode failed on attempt {attempt+1}. Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(1)
            
    # Fallback response if all retries fail
    print("All API retries failed. Using local mock generator fallback for this row.")
    return generate_mock_enrichment(row)

def enrich_dataset(clean_csv_path, enriched_csv_path, limit=100, model="llama-3.1-8b-instant"):
    """
    Enriches clean properties with LLM-generated descriptions, reviews, and amenities.
    Includes checkpointing to resume from previous run.
    """
    global is_mock_mode
    
    client = None
    if not is_mock_mode:
        client = Groq(api_key=api_key)
        
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Clean CSV not found at {clean_csv_path}. Please run preprocess.py first.")
        
    df_clean = pd.read_csv(clean_csv_path)
    
    # Check if enriched CSV already exists for checkpointing
    if os.path.exists(enriched_csv_path):
        print(f"Found existing enriched file: {enriched_csv_path}")
        df_enriched = pd.read_csv(enriched_csv_path)
        processed_ids = set(df_enriched['Listing_Id'].tolist())
        print(f"Already processed {len(processed_ids)} properties.")
    else:
        df_enriched = pd.DataFrame()
        processed_ids = set()
        
    # Get listings that need to be processed
    to_process = df_clean[~df_clean['Listing_Id'].isin(processed_ids)].copy()
    
    if len(to_process) == 0:
        print("All listings are already enriched!")
        return
        
    # Limit number of listings to process in this run
    if limit and limit > 0:
        to_process = to_process.head(limit)
        print(f"Limiting this run to process {limit} properties.")
        
    print(f"Beginning enrichment of {len(to_process)} properties...")
    
    rows_list = []
    
    for idx, (_, row) in enumerate(to_process.iterrows(), 1):
        mode_label = "MOCK" if is_mock_mode else "GROQ"
        print(f"[{idx}/{len(to_process)}] Enriching Listing ID: {row['Listing_Id']} ({row['BHK']} BHK, {row['Area Locality']}, {row['City']}) [{mode_label}]")
        
        enrichment = call_groq_enrichment(client, row, model=model)
        
        # Merge structured row data with synthetic content
        enriched_row = row.to_dict()
        enriched_row['Description'] = enrichment.get('description', '')
        enriched_row['Review'] = enrichment.get('review', '')
        enriched_row['Amenities'] = json.dumps(enrichment.get('amenities', []))
        
        rows_list.append(enriched_row)
        
        # Save checkpoint periodically (every 5 properties or immediately in mock mode if batch is small)
        if idx % 5 == 0 or is_mock_mode:
            if rows_list:
                df_batch = pd.DataFrame(rows_list)
                if not df_enriched.empty:
                    # Filter out existing processed rows to prevent duplicates in merge
                    df_enriched = df_enriched[~df_enriched['Listing_Id'].isin(df_batch['Listing_Id'])]
                    df_enriched = pd.concat([df_enriched, df_batch], ignore_index=True)
                else:
                    df_enriched = df_batch
                
                # Deduplicate df_enriched by Listing_Id just in case
                df_enriched = df_enriched.drop_duplicates(subset=['Listing_Id'])
                # Sort by Listing_Id
                df_enriched = df_enriched.sort_values('Listing_Id')
                
                df_enriched.to_csv(enriched_csv_path, index=False)
                rows_list = [] # Reset batch list
                
            if idx % 5 == 0:
                print(f"--- Saved checkpoint at property {idx} to {enriched_csv_path} ---")
            
        # Mild throttle to avoid immediate rate limits
        if not is_mock_mode:
            time.sleep(0.5)
        
    # Append any remaining rows not saved in the last batch
    if rows_list:
        df_batch = pd.DataFrame(rows_list)
        if not df_enriched.empty:
            df_enriched = df_enriched[~df_enriched['Listing_Id'].isin(df_batch['Listing_Id'])]
            df_enriched = pd.concat([df_enriched, df_batch], ignore_index=True)
        else:
            df_enriched = df_batch
            
        df_enriched = df_enriched.drop_duplicates(subset=['Listing_Id'])
        df_enriched = df_enriched.sort_values('Listing_Id')
        df_enriched.to_csv(enriched_csv_path, index=False)
        
    print(f"Enrichment completed. Saved all outputs to {enriched_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich clean housing dataset with LLM metadata.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of properties to process. Default is 10.")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Groq model to use.")
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv = os.path.join(current_dir, "../data/house_rent_clean.csv")
    enriched_csv = os.path.join(current_dir, "../data/house_rent_enriched.csv")
    
    enrich_dataset(clean_csv, enriched_csv, limit=args.limit, model=args.model)
