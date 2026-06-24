import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, GroqError

# Load environment variables
load_dotenv()

# Define API key
api_key = os.getenv("GROQ_API_KEY")

def generate_mock_recommendations(shortlist_df, soft_preferences):
    """
    Generates realistic fallback recommendations locally when Groq API is unavailable.
    """
    recommendations = []
    
    # Lowercase preferences for basic keyword matching
    pref_lower = str(soft_preferences).lower()
    
    for idx, (_, row) in enumerate(shortlist_df.iterrows(), 1):
        listing_id = int(row['Listing_Id'])
        rent = row['Rent']
        locality = row['Area Locality']
        city = row['City']
        bhk = row['BHK']
        
        # Load synthetic amenities
        try:
            amenities = json.loads(row['Amenities'])
        except Exception:
            amenities = []
            
        # Basic matching logic for explanations
        matched_features = []
        for am in amenities:
            if am.lower() in pref_lower:
                matched_features.append(am)
                
        # Generate custom explanation based on matches
        if matched_features:
            explanation = f"This property is highly recommended because it offers your requested amenities: {', '.join(matched_features)}. Located in the peaceful {locality} area of {city}, it matches your search requirements."
        else:
            explanation = f"Located in {locality}, {city}, this {bhk} BHK home is a strong fit. It is well-situated and offers good value for a rent of INR {rent}."
            
        compromises = ""
        # Check for potential budget compromises
        if rent > 35000:
            compromises = "This property is on the higher end of the typical rent spectrum, which may stretch your budget."
        elif "transit" in pref_lower or "metro" in pref_lower:
            compromises = "While the locality is great, you may need to walk a bit to reach the nearest major transit hub."
        else:
            compromises = "No major compromises detected for this listing."
            
        recommendations.append({
            "listing_id": listing_id,
            "rank": idx,
            "explanation": explanation,
            "compromises": compromises
        })
        
    return {"recommendations": recommendations}

def get_recommendations(shortlist_df, soft_preferences, model="llama-3.1-8b-instant"):
    """
    Uses Groq to reason over soft preferences and rank/explain shortlisted properties.
    """
    # Check if we should fallback to mock immediately
    if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
        print("[WARNING] Groq API Key missing or placeholder. Running in mock recommender mode.")
        return generate_mock_recommendations(shortlist_df, soft_preferences)
        
    if shortlist_df.empty:
        return {"recommendations": []}
        
    client = Groq(api_key=api_key)
    
    # Format shortlisted properties for the prompt
    properties_context = []
    for _, row in shortlist_df.iterrows():
        try:
            amenities_list = json.loads(row['Amenities'])
        except Exception:
            amenities_list = []
            
        prop_str = f"""
        Listing ID: {row['Listing_Id']}
        City: {row['City']}
        Locality: {row['Area Locality']}
        Rent: INR {row['Rent']} per month
        BHK: {row['BHK']} BHK
        Furnishing: {row['Furnishing Status']}
        Size: {row['Size']} Sq. Ft.
        Bathrooms: {row['Bathroom']}
        Description: {row['Description']}
        Tenant Review: {row['Review']}
        Amenities: {', '.join(amenities_list)}
        """
        properties_context.append(prop_str.strip())
        
    properties_formatted = "\n---\n".join(properties_context)
    
    prompt = f"""
    You are an expert real estate consultant for the Indian housing market.
    
    Your task is to rank the shortlisted rental properties below based on how well they fit the user's soft preferences.
    
    User Soft Preferences:
    '''
    {soft_preferences}
    '''
    
    Shortlisted Properties:
    ---
    {properties_formatted}
    ---
    
    Rank all provided properties from best fit to lowest fit.
    You must output a raw JSON object with a single top-level key "recommendations", containing a list of objects.
    Each object in the list must have these exact keys:
    - "listing_id": (integer) The Listing ID of the property.
    - "rank": (integer) The rank assigned (1 is best, followed by 2, 3, etc.).
    - "explanation": (string) A detailed 2-3 sentence personalized explanation of why this property fits (or doesn't fit) the user's soft preferences. Refer to the locality, amenities, or reviews in your explanation.
    - "compromises": (string) 1 sentence explaining any compromises or trade-offs for this property relative to their preferences (e.g. rent cost, size, or lack of specific features).
    
    Do not output markdown code blocks or explanations outside of the raw JSON object.
    """
    
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
        print(f"[ERROR] Groq API call failed: {e}. Falling back to mock recommendations.")
        return generate_mock_recommendations(shortlist_df, soft_preferences)
    except Exception as e:
        print(f"[ERROR] Parsing recommendations failed: {e}. Falling back to mock recommendations.")
        return generate_mock_recommendations(shortlist_df, soft_preferences)
