import os
import json
import pandas as pd
import streamlit as st
from src.filter import apply_hard_filters, score_and_shortlist
from src.recommender import get_recommendations
from src.enrich import generate_mock_enrichment

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="RentAI - Personalized Property Matchmaker",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Datasets
@st.cache_data
def load_base_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(current_dir, "data/house_rent_clean.csv")
    enriched_path = os.path.join(current_dir, "data/house_rent_enriched.csv")
    
    df_clean = pd.read_csv(clean_path) if os.path.exists(clean_path) else pd.DataFrame()
    df_enriched = pd.read_csv(enriched_path) if os.path.exists(enriched_path) else pd.DataFrame()
    
    # Merge clean data and enriched data so all cities (Mumbai, Bangalore, etc.) are available,
    # with Description, Review, and Amenities columns pre-populated for Kolkata
    if not df_clean.empty:
        if not df_enriched.empty:
            enriched_cols = df_enriched[['Listing_Id', 'Description', 'Review', 'Amenities']]
            working_df = pd.merge(df_clean, enriched_cols, on='Listing_Id', how='left')
        else:
            working_df = df_clean.copy()
    else:
        working_df = df_enriched.copy()
        
    return working_df

df_working = load_base_datasets()

# Inject Premium CSS Style (Clean & Minimalist Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Font Setup */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Layout Background overrides */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* Custom Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* Main Header Styling */
    .app-header {
        text-align: center;
        padding: 25px 0;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 38px;
        font-weight: 800;
        color: #4F46E5;
        margin-bottom: 6px;
    }
    .app-subtitle {
        font-size: 15px;
        color: #475569;
    }
    
    /* Clean Minimalist Card Style */
    .property-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    }
    .property-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border-color: #6366F1;
    }
    
    /* Text Styles inside Card */
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .card-location {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    /* Harmonious Tags Styling */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }
    .badge {
        font-size: 13px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 30px;
    }
    .badge-bhk {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4338CA;
    }
    .badge-size {
        background: #FEF2F2;
        border: 1px solid #FEE2E2;
        color: #991B1B;
    }
    .badge-furnishing {
        background: #FFFBEB;
        border: 1px solid #FEF3C7;
        color: #92400E;
    }
    .badge-bath {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        color: #15803D;
    }
    .badge-rent {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4338CA;
        font-size: 15px;
        font-weight: 700;
    }
    
    /* AI Explanation Styling (Indigo left-border accent box) */
    .ai-explanation-box {
        background: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 16px;
    }
    .ai-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #4F46E5;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .ai-text {
        font-size: 14px;
        line-height: 1.5;
        color: #334155;
    }
    .ai-compromises {
        font-size: 13px;
        color: #BE123C;
        margin-top: 8px;
        font-style: italic;
    }
    
    /* Amenities List */
    .amenities-title {
        font-size: 13px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    .amenity-chip {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        color: #475569;
    }

    /* Disable typing/search in selectboxes to make them dropdown-only */
    div[data-testid="stSelectbox"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
        cursor: pointer !important;
    }

    /* Clean Sidebar styling to match the main content panel */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] label {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-bottom: 6px !important;
        margin-top: 12px !important;
    }

    [data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] > div,
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] div,
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] div,
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] span,
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] span {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] > div {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSelectbox"] input,
    div[data-testid="stMultiSelect"] input {
        color: #0F172A !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }

    /* Style multiselect input tags to look like our badges */
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] span[data-baseweb="tag"] * {
        background-color: transparent !important;
        color: inherit !important;
    }

    /* Style the range slider handles and track to match the Indigo brand color */
    [data-testid="stSidebar"] div[role="slider"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSliderTrack"] > div {
        background-color: #4F46E5 !important;
    }
    
    /* Search button inside sidebar styled as a beautiful Indigo primary button */
    [data-testid="stSidebar"] button {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
        margin-top: 15px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #4338CA !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3) !important;
        color: #FFFFFF !important;
        border-color: #4338CA !important;
    }

    /* Style all virtual dropdown menu popovers to have a light theme */
    [data-testid*="Dropdown"],
    [data-testid*="dropdown"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1) !important;
    }

    /* Style option items inside dropdown lists */
    [data-testid*="Dropdown"] li,
    [data-testid*="dropdown"] li,
    [data-testid*="Dropdown"] div[role="option"],
    [data-testid*="dropdown"] div[role="option"],
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] div,
    div[role="listbox"] div,
    ul[role="listbox"] li {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    /* Reset background and text color for all child elements inside dropdowns to inherit from parent option */
    div[data-baseweb="popover"] *,
    [data-testid*="Dropdown"] * {
        background-color: transparent !important;
        color: inherit !important;
    }

    /* Option item hover and selection states */
    [data-testid*="Dropdown"] li:hover,
    [data-testid*="dropdown"] li:hover,
    [data-testid*="Dropdown"] div[role="option"]:hover,
    [data-testid*="dropdown"] div[role="option"]:hover,
    [data-testid*="Dropdown"] li[aria-selected="true"],
    [data-testid*="dropdown"] li[aria-selected="true"],
    [data-testid*="Dropdown"] div[role="option"][aria-selected="true"],
    [data-testid*="dropdown"] div[role="option"][aria-selected="true"],
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] div:hover,
    div[role="listbox"] div:hover,
    ul[role="listbox"] li:hover {
        background-color: #F1F5F9 !important;
        color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Title
st.markdown("""
<div class="app-header">
    <div class="app-title">RentAI Property Matchmaker</div>
    <div class="app-subtitle">AI-Powered Rental Property Recommendations for the Indian Housing Market</div>
</div>
""", unsafe_allow_html=True)

# Determine the list of cities based on raw data
all_cities = sorted(df_working['City'].unique()) if not df_working.empty else ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"]

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.markdown('<h3 style="font-size: 18px; font-weight: 700; color: #0F172A; margin-top: 10px; margin-bottom: 15px; font-family: \'Plus Jakarta Sans\', sans-serif;">🏠 Hard Criteria Filters</h3>', unsafe_allow_html=True)

# City selection
selected_city = st.sidebar.selectbox("Select Target City", all_cities, index=0)

# Rent Range selection
max_val = int(df_working[df_working['City'] == selected_city]['Rent'].max()) if not df_working.empty else 150000
min_val = int(df_working[df_working['City'] == selected_city]['Rent'].min()) if not df_working.empty else 2000
rent_range = st.sidebar.slider(
    "Monthly Rent Budget (INR)",
    min_value=min_val,
    max_value=min(max_val, 200000), 
    value=(min_val, min(max_val, 30000)),
    step=500
)

# BHK configuration (multi-select)
bhk_options = ["1 BHK", "2 BHK", "3 BHK", "4+ BHK"]
selected_bhks = st.sidebar.multiselect("BHK Configuration", bhk_options, default=["2 BHK"])

# Furnishing Status (multi-select)
furnishing_options = ["Furnished", "Semi-Furnished", "Unfurnished"]
selected_furnishing = st.sidebar.multiselect("Furnishing Status", furnishing_options, default=furnishing_options)

# Tenant Preferred
tenant_type = st.sidebar.selectbox("Your Tenant Profile", ["Bachelors/Family", "Bachelors", "Family"], index=0)

st.sidebar.markdown('<hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 20px 0;" />', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 15px; font-family: \'Plus Jakarta Sans\', sans-serif;">✨ AI Soft Preferences</h3>', unsafe_allow_html=True)
soft_preferences = st.sidebar.text_area(
    "Describe your dream home vibe...",
    placeholder="e.g. Needs to be peaceful, spacious, close to a park, with good security, high-speed internet, and a nice balcony for morning coffee.",
    help="AI will reason over this description to rank and explain your recommended listings.",
    height=120
)

search_button = st.sidebar.button("🔍 Find My Match", use_container_width=True)

# ----------------- RESULTS RENDERING -----------------
if search_button or 'first_run' not in st.session_state:
    st.session_state['first_run'] = True
    
    # 1. Filter dataset programmatically
    working_df = df_working.copy()
    
    # Map visual BHK selections back to numbers
    bhk_mapping = []
    for bhk in selected_bhks:
        if bhk == "4+ BHK":
            bhk_mapping.append("4+")
        else:
            bhk_mapping.append(int(bhk.split()[0]))
            
    filtered_df = apply_hard_filters(
        working_df,
        city=selected_city,
        min_rent=rent_range[0],
        max_rent=rent_range[1],
        bhk_list=bhk_mapping,
        furnishing_list=selected_furnishing,
        tenant_type=tenant_type
    )
    
    # 2. Handle empty results case
    if filtered_df.empty:
        st.markdown(f"""
        <div style="background: #FFF2F0; border: 1px solid #FFE0DB; padding: 20px; border-radius: 12px; margin-top: 10px; font-family: 'Plus Jakarta Sans', sans-serif;">
            <h4 style="color: #BE123C; margin-top:0; font-weight: 700;">⚠️ No exact matches found</h4>
            <p style="color: #475569; margin-bottom: 0;">We couldn't find any properties matching your exact filters in <strong>{selected_city}</strong> within INR {rent_range[0]:,} - {rent_range[1]:,}. Try widening your budget or selecting more furnishing options.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # fallback suggest wider matches
        fallback_df = apply_hard_filters(
            working_df,
            city=selected_city,
            min_rent=int(rent_range[0] * 0.8),
            max_rent=int(rent_range[1] * 1.2),
            bhk_list=bhk_mapping,
            furnishing_list=furnishing_options,
            tenant_type=tenant_type
        )
        if not fallback_df.empty:
            st.markdown("<h3 style='font-size: 20px; font-weight: 700; color: #0F172A; margin-top: 25px;'>💡 Recommended Alternatives (with relaxed constraints):</h3>", unsafe_allow_html=True)
            shortlist_fallback = score_and_shortlist(fallback_df, min_rent=rent_range[0], max_rent=rent_range[1], top_n=3)
            # Make sure fallback rows have enrichment fields
            for idx, row in shortlist_fallback.iterrows():
                if 'Description' not in shortlist_fallback.columns or pd.isna(row.get('Description')):
                    mock_enrich = generate_mock_enrichment(row)
                    shortlist_fallback.at[idx, 'Description'] = mock_enrich['description']
                    shortlist_fallback.at[idx, 'Review'] = mock_enrich['review']
                    shortlist_fallback.at[idx, 'Amenities'] = json.dumps(mock_enrich['amenities'])
            
            # Simple fallback recommendations
            recs = get_recommendations(shortlist_fallback, soft_preferences)
            rec_map = {rec['listing_id']: rec for rec in recs.get('recommendations', [])}
            
            for _, row in shortlist_fallback.iterrows():
                l_id = int(row['Listing_Id'])
                r_details = rec_map.get(l_id, {"explanation": "Matches closely with your alternative budget parameters.", "compromises": "Rent limits slightly relaxed."})
                try:
                    amenities_list = json.loads(row['Amenities'])
                except Exception:
                    amenities_list = []
                    
                st.markdown(f"""
                <div class="property-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div class="card-title">{row['BHK']} BHK Apartment in {row['Area Locality']}</div>
                            <div class="card-location">📍 {row['City']} • {row['Area Type']} • Preferred: {row['Tenant Preferred']}</div>
                        </div>
                        <div class="badge badge-rent">₹{row['Rent']:,}/mo</div>
                    </div>
                    <div class="tag-container">
                        <span class="badge badge-bhk">🛏️ {row['BHK']} BHK</span>
                        <span class="badge badge-size">📏 {row['Size']} Sq.Ft.</span>
                        <span class="badge badge-furnishing">🛋️ {row['Furnishing Status']}</span>
                    </div>
                    <div style="color: #334155; font-size: 14px; margin-bottom: 12px;">{row['Description']}</div>
                    <div class="amenities-title">Amenities:</div>
                    <div style="margin-bottom: 12px;">{" ".join([f'<span class="amenity-chip">{am}</span>' for am in amenities_list])}</div>
                    <div class="ai-explanation-box">
                        <div class="ai-label">🤖 AI Fit Analysis</div>
                        <div class="ai-text">{r_details.get('explanation')}</div>
                        <div class="ai-compromises">⚠️ Compromise: {r_details.get('compromises')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        # 3. Shortlist candidates
        st.markdown(f"Found **{len(filtered_df)}** matching properties. Shortlisting top candidates for AI analysis...")
        shortlist_df = score_and_shortlist(filtered_df, min_rent=rent_range[0], max_rent=rent_range[1], top_n=5)
        
        # Ensure shortlisted columns have Descriptions, Reviews, and Amenities
        # (Needed if city was clean but not batch pre-enriched)
        for idx, row in shortlist_df.iterrows():
            if 'Description' not in shortlist_df.columns or pd.isna(row.get('Description')) or str(row.get('Description')).strip() == "":
                mock_enrich = generate_mock_enrichment(row)
                
                # Check and add columns if they don't exist
                if 'Description' not in shortlist_df.columns:
                    shortlist_df['Description'] = ""
                if 'Review' not in shortlist_df.columns:
                    shortlist_df['Review'] = ""
                if 'Amenities' not in shortlist_df.columns:
                    shortlist_df['Amenities'] = ""
                    
                shortlist_df.at[idx, 'Description'] = mock_enrich['description']
                shortlist_df.at[idx, 'Review'] = mock_enrich['review']
                shortlist_df.at[idx, 'Amenities'] = json.dumps(mock_enrich['amenities'])
        
        # 4. Generate recommendations using Groq / Fallback
        with st.spinner("🤖 AI is analyzing property vibes and ranking matches..."):
            recommendations_data = get_recommendations(shortlist_df, soft_preferences)
            
        recommendations = recommendations_data.get('recommendations', [])
        
        # Map recommendations by listing ID
        rec_map = {rec['listing_id']: rec for rec in recommendations}
        
        # Sort shortlist_df based on the AI rank
        shortlist_df['ai_rank'] = shortlist_df['Listing_Id'].map(lambda x: rec_map.get(int(x), {}).get('rank', 99))
        shortlist_df = shortlist_df.sort_values('ai_rank')
        
        # Render Properties
        st.markdown("<h2 style='font-size: 28px; font-weight: 800; color: #0F172A; margin-top: 20px;'>🎯 Top Property Matches</h2>", unsafe_allow_html=True)
        
        for idx, row in shortlist_df.iterrows():
            l_id = int(row['Listing_Id'])
            rec_info = rec_map.get(l_id, {})
            rank = rec_info.get('rank', idx + 1)
            explanation = rec_info.get('explanation', "Matches your search criteria and offers great value.")
            compromises = rec_info.get('compromises', "No major compromises detected.")
            
            try:
                amenities_list = json.loads(row['Amenities'])
            except Exception:
                amenities_list = []
                
            st.markdown(f"""
            <div class="property-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <div class="card-title">#{rank} | {row['BHK']} BHK Apartment in {row['Area Locality']}</div>
                        <div class="card-location">📍 {row['City']} • {row['Area Type']} • Preferred Tenant: {row['Tenant Preferred']}</div>
                    </div>
                    <div class="badge badge-rent">₹{row['Rent']:,}/mo</div>
                </div>
                <div class="tag-container">
                    <span class="badge badge-bhk">🛌 {row['BHK']} BHK</span>
                    <span class="badge badge-size">📐 {row['Size']} Sq.Ft.</span>
                    <span class="badge badge-furnishing">🛋️ {row['Furnishing Status']}</span>
                    <span class="badge badge-bath">🚿 {row['Bathroom']} Bath</span>
                </div>
                <div style="margin-bottom: 12px; color: #334155; font-size: 14px; line-height: 1.5;">
                    <strong>Description:</strong> {row['Description']}
                </div>
                <div class="amenities-title">Amenities:</div>
                <div style="margin-bottom: 12px;">
                    {" ".join([f'<span class="amenity-chip">{am}</span>' for am in amenities_list])}
                </div>
                <div style="margin-bottom: 12px; font-size: 13px; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 8px; font-style: italic;">
                    <strong>Past Tenant Feedback:</strong> "{row['Review']}"
                </div>
                <div class="ai-explanation-box">
                    <div class="ai-label">🤖 AI Fit Analysis</div>
                    <div class="ai-text">{explanation}</div>
                    <div class="ai-compromises">⚠️ Trade-off: {compromises}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
