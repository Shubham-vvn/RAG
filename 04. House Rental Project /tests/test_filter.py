import pytest
import pandas as pd
from src.filter import apply_hard_filters, score_and_shortlist

@pytest.fixture
def mock_df():
    data = [
        {
            "Listing_Id": 1,
            "City": "Kolkata",
            "Rent": 10000,
            "BHK": 2,
            "Furnishing Status": "Unfurnished",
            "Tenant Preferred": "Bachelors/Family",
            "Size": 1000,
            "Amenities": '["Water Supply", "Exhaust Fan"]'
        },
        {
            "Listing_Id": 2,
            "City": "Kolkata",
            "Rent": 20000,
            "BHK": 2,
            "Furnishing Status": "Semi-Furnished",
            "Tenant Preferred": "Family",
            "Size": 800,
            "Amenities": '["Water Supply", "Geyser"]'
        },
        {
            "Listing_Id": 3,
            "City": "Mumbai",
            "Rent": 35000,
            "BHK": 3,
            "Furnishing Status": "Furnished",
            "Tenant Preferred": "Bachelors",
            "Size": 1200,
            "Amenities": '["Water Supply", "Gym", "Pool"]'
        },
        {
            "Listing_Id": 4,
            "City": "Mumbai",
            "Rent": 8000,
            "BHK": 1,
            "Furnishing Status": "Unfurnished",
            "Tenant Preferred": "Bachelors/Family",
            "Size": 300,
            "Amenities": '["Water Supply"]'
        }
    ]
    return pd.DataFrame(data)

def test_apply_hard_filters_city(mock_df):
    res = apply_hard_filters(mock_df, city="Kolkata")
    assert len(res) == 2
    assert all(res['City'] == "Kolkata")

def test_apply_hard_filters_rent(mock_df):
    res = apply_hard_filters(mock_df, min_rent=15000, max_rent=40000)
    assert len(res) == 2
    assert sorted(res['Listing_Id'].tolist()) == [2, 3]

def test_apply_hard_filters_bhk(mock_df):
    res = apply_hard_filters(mock_df, bhk_list=[3])
    assert len(res) == 1
    assert res.iloc[0]['Listing_Id'] == 3

def test_apply_hard_filters_furnishing(mock_df):
    res = apply_hard_filters(mock_df, furnishing_list=["Furnished", "Semi-Furnished"])
    assert len(res) == 2
    assert sorted(res['Listing_Id'].tolist()) == [2, 3]

def test_apply_hard_filters_tenant_bachelors(mock_df):
    # Bachelors matches Bachelors or Bachelors/Family
    res = apply_hard_filters(mock_df, tenant_type="Bachelors")
    assert len(res) == 3
    assert sorted(res['Listing_Id'].tolist()) == [1, 3, 4]

def test_apply_hard_filters_tenant_family(mock_df):
    # Family matches Family or Bachelors/Family
    res = apply_hard_filters(mock_df, tenant_type="Family")
    assert len(res) == 3
    assert sorted(res['Listing_Id'].tolist()) == [1, 2, 4]

def test_score_and_shortlist(mock_df):
    # Test shortlisting logic and ordering
    res = score_and_shortlist(mock_df, min_rent=5000, max_rent=40000, top_n=2)
    assert len(res) == 2
    # Verify it doesn't crash and returns the requested number of properties
