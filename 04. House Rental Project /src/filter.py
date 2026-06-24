import os
import pandas as pd

def apply_hard_filters(df, city=None, min_rent=0, max_rent=float('inf'), bhk_list=None, furnishing_list=None, tenant_type=None):
    """
    Applies programmatic hard filters to the housing dataframe.
    """
    filtered_df = df.copy()
    
    # 1. Filter by City (case-insensitive)
    if city:
        city_clean = str(city).strip().lower()
        filtered_df = filtered_df[filtered_df['City'].str.strip().str.lower() == city_clean]
        
    # 2. Filter by Rent Range
    filtered_df = filtered_df[(filtered_df['Rent'] >= min_rent) & (filtered_df['Rent'] <= max_rent)]
    
    # 3. Filter by BHK configuration
    if bhk_list:
        # Convert all to integers and handle 4+ BHK
        clean_bhks = []
        has_four_plus = False
        for b in bhk_list:
            if str(b).strip().endswith('+') or (isinstance(b, str) and '4' in b and '+' in b):
                has_four_plus = True
            else:
                try:
                    clean_bhks.append(int(b))
                except ValueError:
                    pass
        
        if has_four_plus:
            # If 4+ BHK is chosen, we match anything >= 4 OR any of the specific smaller BHKs
            if clean_bhks:
                filtered_df = filtered_df[(filtered_df['BHK'].isin(clean_bhks)) | (filtered_df['BHK'] >= 4)]
            else:
                filtered_df = filtered_df[filtered_df['BHK'] >= 4]
        else:
            if clean_bhks:
                filtered_df = filtered_df[filtered_df['BHK'].isin(clean_bhks)]
                
    # 4. Filter by Furnishing Status
    if furnishing_list:
        furnishing_clean = [str(f).strip().lower() for f in furnishing_list]
        filtered_df = filtered_df[filtered_df['Furnishing Status'].str.strip().str.lower().isin(furnishing_clean)]
        
    # 5. Filter by Tenant Preference
    # Dataset values: "Bachelors/Family", "Bachelors", "Family"
    if tenant_type:
        tenant_clean = str(tenant_type).strip().lower()
        # If user is a Bachelor, match "Bachelors" or "Bachelors/Family"
        if "bachelor" in tenant_clean:
            filtered_df = filtered_df[filtered_df['Tenant Preferred'].str.strip().str.lower().isin(['bachelors', 'bachelors/family'])]
        # If user is a Family, match "Family" or "Bachelors/Family"
        elif "family" in tenant_clean:
            filtered_df = filtered_df[filtered_df['Tenant Preferred'].str.strip().str.lower().isin(['family', 'bachelors/family'])]
        # Otherwise match everything (no hard restriction for company)
        
    return filtered_df.reset_index(drop=True)

def score_and_shortlist(df_filtered, min_rent=0, max_rent=100000, top_n=5):
    """
    Selects the best top_n candidates from the filtered dataframe
    using value score (Size / Rent) and budget proximity.
    """
    if df_filtered.empty:
        return df_filtered
        
    df = df_filtered.copy()
    
    # Calculate budget midpoint proximity score
    mid_rent = (min_rent + max_rent) / 2 if max_rent != float('inf') else min_rent * 1.5
    if mid_rent <= 0:
        mid_rent = 1 # Avoid division by zero
        
    df['budget_proximity'] = 1 / (1 + (df['Rent'] - mid_rent).abs() / mid_rent)
    
    # Value score: space per unit cost
    df['size_value'] = df['Size'] / df['Rent']
    # Normalize size value to range [0, 1]
    max_val = df['size_value'].max()
    min_val = df['size_value'].min()
    if max_val != min_val:
        df['size_value_norm'] = (df['size_value'] - min_val) / (max_val - min_val)
    else:
        df['size_value_norm'] = 1.0
        
    # Combined score rank: 60% budget proximity, 40% value
    df['rank_score'] = df['budget_proximity'] * 0.6 + df['size_value_norm'] * 0.4
    
    # Return top_n
    shortlisted = df.nlargest(top_n, 'rank_score')
    return shortlisted.reset_index(drop=True)
