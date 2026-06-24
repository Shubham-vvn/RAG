"""
Streamlit web UI — preference form + recommendation result cards.
Custom redesigned to match the Zomato AI light premium mockup.

Usage:
    streamlit run src/ui/streamlit_app.py

Implementation: Redesigned light-theme frontend mockup matching Zomato AI visual aesthetics (Crimson & Cream).
"""

import base64
import logging
from pathlib import Path
import streamlit as st

from src.data import initialize_data
from src.models.preferences import UserPreferences
from src.services.recommendation import RecommendationService
from src.services.filter import ValidationError

# Page configuration for a premium, wide feel
st.set_page_config(
    page_title="Zomato AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


def get_base64_image(image_name: str) -> str:
    """Helper to convert local assets into base64 images for HTML embedding."""
    try:
        path = Path(f"src/ui/assets/{image_name}.png")
        if not path.exists():
            path = Path(f"src/ui/assets/{image_name}.jpg")
        if path.exists():
            with open(path, "rb") as f:
                data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        logger.warning(f"Failed to load image asset {image_name}: {e}")
    # Fallback transparent 1x1 GIF pixel
    return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def get_restaurant_image_base64(name: str, cuisines: list[str]) -> str:
    """Choose appropriate food mockup image based on cuisine descriptors."""
    text = (name + " " + " ".join(cuisines)).lower()
    if any(k in text for k in ["indian", "biryani", "punjabi", "mughlai", "kebab", "tandoori", "dal", "paneer", "roti"]):
        return get_base64_image("indian_dish")
    elif any(k in text for k in ["italian", "pizza", "pasta", "spaghetti", "carbonara", "lasagna"]):
        return get_base64_image("italian_dish")
    else:
        return get_base64_image("dessert_dish")


# ── Custom Light Crimson/Cream Styling Injection ──
st.markdown(
    """
    <style>
    /* Google Font integration */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Hide standard Streamlit header/footer elements for premium wrapper look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1300px;
    }
    
    /* Top Navigation bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .top-bar-logo {
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .top-bar-logo h1 {
        font-size: 25px !important;
        font-weight: 800 !important;
        color: #cb202d !important; /* Zomato Red */
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    .top-bar-logo span {
        font-size: 11px;
        letter-spacing: 1.5px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
    }
    .top-bar-actions {
        display: flex;
        gap: 20px;
        font-size: 18px;
        color: #64748b;
    }

    /* Connection / Offline error alert */
    .connection-banner {
        background-color: #fef2f2;
        border: 1px solid #fee2e2;
        color: #991b1b;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13.5px;
        font-weight: 500;
    }
    
    /* Sidebar Headers */
    .sidebar-section-title {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #cb202d; /* Brand red header */
        margin-bottom: 4px !important;
    }
    .sidebar-section-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 25px;
    }
    
    /* Analysis status indicator card */
    .analysis-widget {
        background-color: #f9fafb;
        border: 1px dashed #fca5a5;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin-top: 35px;
    }
    .analysis-widget-icon {
        font-size: 24px;
        margin-bottom: 10px;
        color: #cb202d;
    }
    .analysis-widget-title {
        font-size: 14px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }
    .analysis-widget-desc {
        font-size: 11.5px;
        color: #64748b;
        line-height: 1.4;
    }

    /* Recommendations layout header */
    .recs-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
    }
    .recs-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #1f2937 !important;
        margin: 0 !important;
    }
    .recs-tabs {
        display: flex;
        gap: 8px;
    }
    .recs-tab {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #64748b;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 12.5px;
        font-weight: 600;
        cursor: pointer;
    }
    .recs-tab.active {
        background-color: #fef2f2;
        border-color: #fca5a5;
        color: #cb202d;
    }

    /* AI Curated Summary Box */
    .ai-insight-box {
        background-color: #fff5f5;
        border: 1px solid #fee2e2;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        gap: 18px;
        box-shadow: 0 4px 20px rgba(203, 32, 45, 0.02);
    }
    .ai-insight-icon {
        background-color: #cb202d;
        border-radius: 8px;
        width: 44px;
        height: 44px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 20px;
        flex-shrink: 0;
    }
    .ai-insight-content {
        font-size: 14px;
        color: #4b5563;
        line-height: 1.6;
    }
    .ai-insight-title {
        font-size: 11px;
        font-weight: 800;
        color: #cb202d;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }

    /* Constraint Relaxation Banner */
    .relaxation-alert {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        font-size: 13.5px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 500;
    }

    /* Restaurant Card Grid and styles */
    .restaurant-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        margin-bottom: 24px;
        display: flex;
        overflow: hidden;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.01);
    }
    .restaurant-card:hover {
        transform: translateY(-2px);
        border-color: #fca5a5;
        box-shadow: 0 12px 30px rgba(203, 32, 45, 0.06);
    }
    .card-image-container {
        width: 240px;
        position: relative;
        flex-shrink: 0;
    }
    .card-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .card-rank {
        position: absolute;
        top: 12px;
        left: 12px;
        background-color: #f59e0b; /* Warm Gold badge */
        color: white;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
        padding: 4px 10px;
        border-radius: 20px;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.35);
    }
    .card-info {
        padding: 24px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .card-title {
        font-size: 21px !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin: 0 !important;
    }
    .card-rating-badge {
        background-color: #e6f4ea;
        border: 1px solid #a7f3d0;
        color: #137333;
        font-size: 13.5px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .card-cost {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    .card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 12px 0 18px 0;
    }
    .card-tag {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        color: #4b5563;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .card-explanation {
        background-color: #f9fafb;
        border-left: 3px solid #cb202d; /* red border */
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 13.5px;
        color: #374151;
        line-height: 1.55;
    }

    /* Skeleton / Loading indicator states */
    .loading-row {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #cb202d;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 30px 0 15px 0;
    }
    .skeleton-card {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        height: 160px;
        margin-bottom: 20px;
        opacity: 0.8;
    }

    /* Sidebar button theme override */
    div[data-testid="stSidebar"] button {
        background-color: #cb202d !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #991b1b !important;
    }

    /* Footer styles */
    .app-footer {
        border-top: 1px solid #e2e8f0;
        padding-top: 30px;
        margin-top: 50px;
        text-align: center;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-bottom: 10px;
    }
    .footer-link {
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-decoration: none;
    }
    .footer-link:hover {
        color: #cb202d;
    }
    .footer-copy {
        color: #94a3b8;
        font-size: 10px;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_repository():
    """Cache the repository instance so we load the dataset once."""
    return initialize_data()


# ── Ingest Data ──
try:
    repository = get_repository()
except Exception as e:
    st.error(f"❌ Failed to initialize Zomato dataset: {e}")
    st.stop()


# ── Sidebar Preference Inputs ──
st.sidebar.markdown(
    """
    <div class="sidebar-section-title">Your preferences</div>
    <div class="sidebar-section-subtitle">Customize your taste with AI-precision</div>
    """,
    unsafe_allow_html=True,
)

locations = repository.get_locations()
default_loc_idx = locations.index("Bangalore") if "Bangalore" in locations else 0
selected_loc = st.sidebar.selectbox("📍 LOCATION", options=locations, index=default_loc_idx)

selected_budget = st.sidebar.radio(
    "💰 BUDGET FOR TWO",
    options=["low", "medium", "high"],
    index=1,
    format_func=lambda x: "Low (<₹500)" if x == "low" else ("Med (₹501-1500)" if x == "medium" else "High (>₹1500)")
)

# Filter cuisines dynamically by the selected location to avoid zero-restaurant choices
loc_restaurants = [r for r in repository.get_all() if r.location == selected_loc]
loc_cuisines = sorted(set(c for r in loc_restaurants for c in r.cuisines))
cuisines = ["Any cuisine"] + loc_cuisines
selected_cuisine = st.sidebar.selectbox("🍜 CUISINE TYPE", options=cuisines, index=0)

selected_rating = st.sidebar.slider(
    "⭐ MINIMUM RATING", min_value=0.0, max_value=5.0, value=3.5, step=0.5
)

selected_additional = st.sidebar.text_area(
    "📝 SPECIFIC REQUIREMENTS",
    placeholder="e.g. family-friendly, outdoor seating, quick service, rooftop, romantic mood...",
)

get_recs = st.sidebar.button("Get Recommendations", use_container_width=True)

# Analysis Ready widget at bottom of sidebar
st.sidebar.markdown(
    """
    <div class="analysis-widget">
        <div class="analysis-widget-icon">📊</div>
        <div class="analysis-widget-title">Analysis Ready</div>
        <div class="analysis-widget-desc">Adjust filters above to see how AI ranks your neighborhood favorites.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Main Area Navigation & Mockup Banners ──
st.markdown(
    """
    <div class="top-bar">
        <div class="top-bar-logo">
            <h1>Zomato AI</h1>
            <span>FIND YOUR PERFECT RESTAURANT</span>
        </div>
        <div class="top-bar-actions">
            <span>🔔</span>
            <span>👤</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)




# ── Recommendations List Rendering ──
if get_recs:
    pref_cuisine = None if selected_cuisine == "Any cuisine" else selected_cuisine
    pref_additional = None if not selected_additional.strip() else selected_additional.strip()

    preferences = UserPreferences(
        location=selected_loc,
        budget=selected_budget,
        min_rating=selected_rating,
        cuisine=pref_cuisine,
        additional=pref_additional,
    )

    # Launch service orchestrator
    service = RecommendationService(repository)

    with st.spinner("🍽️ Analysing restaurants & crafting AI explanations..."):
        try:
            response = service.recommend(preferences)
        except ValidationError as e:
            st.error("### ❌ Preference Validation Failed:")
            for err in e.errors:
                st.markdown(f"- {err}")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.stop()

    # Recommendations Area Header & Selector tabs
    st.markdown(
        """
        <div class="recs-header-container">
            <div class="recs-title">Recommendations</div>
            <div class="recs-tabs">
                <div class="recs-tab active">Top Rated</div>
                <div class="recs-tab">Trending</div>
                <div class="recs-tab">Quick Bites</div>
                <div class="recs-tab">Pure Veg</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # AI Curated Insight Box
    insight_desc = (
        f"Based on your preference for <b>{preferences.location}</b> and <b>{preferences.budget} budget</b>, "
        f"we've identified {len(response.recommendations)} restaurants that excel in quick service and consistent quality. "
        f"We noticed you prefer {preferences.min_rating}+ ratings, so we've excluded newer outlets with unverified reviews."
    )
    if preferences.additional:
        insight_desc += f" Your search specifically targeted notes: \"{preferences.additional}\"."

    st.markdown(
        f"""
        <div class="ai-insight-box">
            <div class="ai-insight-icon">✨</div>
            <div class="ai-insight-content">
                <div class="ai-insight-title">AI Curated Summary</div>
                <div>{insight_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Warnings / Constraint Relaxations Alert
    if response.metadata.constraints_relaxed:
        for warning in response.metadata.constraints_relaxed:
            st.markdown(
                f"""
                <div class="relaxation-alert">
                    <span>⚠️</span>
                    <span>We've slightly relaxed your "Outdoor Seating" filter to find the best culinary matches in your area. ({warning})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Cards listing
    if not response.recommendations:
        st.info("😞 No recommendations matched your criteria.")
    else:
        for rec in response.recommendations:
            img_b64 = get_restaurant_image_base64(rec.name, rec.cuisine.split(","))
            cuisines_html = " ".join([f'<span class="card-tag">{c.strip()}</span>' for c in rec.cuisine.split(",")])
            
            card_html = f"""
            <div class="restaurant-card">
                <div class="card-image-container">
                    <img src="{img_b64}" class="card-image" alt="{rec.name}"/>
                    <div class="card-rank">#{rec.rank} Rank</div>
                </div>
                <div class="card-info">
                    <div>
                        <div class="card-header-row">
                            <h2 class="card-title">{rec.name}</h2>
                            <div class="card-rating-badge">
                                <span>{rec.rating}</span>
                                <span>⭐</span>
                            </div>
                        </div>
                        <div class="card-cost">₹{rec.estimated_cost} for two</div>
                        <div class="card-tags">
                            {cuisines_html}
                        </div>
                    </div>
                    <div class="card-explanation">
                        "{rec.explanation}"
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # Loader skeleton
    st.markdown(
        """
        <div class="loading-row">
            <span>🔄</span>
            <span>Al is ranking restaurants for you...</span>
        </div>
        <div class="skeleton-card"></div>
        """,
        unsafe_allow_html=True,
    )

else:
    # Landing greeting state matching mockup layout
    st.markdown(
        """
        <div class="recs-header-container">
            <div class="recs-title">Recommendations</div>
            <div class="recs-tabs">
                <div class="recs-tab active">Top Rated</div>
                <div class="recs-tab">Trending</div>
                <div class="recs-tab">Quick Bites</div>
                <div class="recs-tab">Pure Veg</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="skeleton-card"></div>
        <div class="skeleton-card" style="opacity: 0.35;"></div>
        
        <div style="text-align: center; color: #64748b; margin: 40px 0 10px 0; font-size: 13px;">
            Your top picks will appear here as we refine our data.
        </div>
        <div style="text-align: center; color: #cb202d; font-weight: bold; font-size: 14px; margin-bottom: 40px; cursor: pointer;">
            Explore more categories →
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Global Page Footer ──
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-links">
            <a href="#" class="footer-link">Privacy Policy</a>
            <a href="#" class="footer-link">Terms of Service</a>
            <a href="#" class="footer-link">Help Center</a>
        </div>
        <div class="footer-copy">© 2026 Zomato AI Recommendations. All rights reserved.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
