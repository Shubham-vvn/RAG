"""
verify_phase1.py  —  Run Phase 1 logic tests without pytest or pydantic.
Usage:  PYTHONPATH=. python3 verify_phase1.py
"""
import sys, types
import pandas as pd

# ── Stub pydantic_settings / pydantic ─────────────────────────────────────
fake_ps = types.ModuleType("pydantic_settings")
class BaseSettings:
    model_config = {}
    def __init__(self, **kw): pass
class SettingsConfigDict(dict): pass
fake_ps.BaseSettings = BaseSettings
fake_ps.SettingsConfigDict = SettingsConfigDict
sys.modules["pydantic_settings"] = fake_ps

fake_p = types.ModuleType("pydantic")
fake_p.Field = lambda *a, **kw: None
fake_p.field_validator = lambda *a, **kw: (lambda f: f)
sys.modules["pydantic"] = fake_p

# ── Import real modules ────────────────────────────────────────────────────
from src.config import SchemaError, DataLoadError          # noqa: E402
from src.data.preprocessor import DataPreprocessor         # noqa: E402
from src.models.restaurant import Restaurant               # noqa: E402
from src.data.repository import RestaurantRepository       # noqa: E402

PP = lambda: DataPreprocessor(budget_thresholds={"low": 500, "medium": 1500})

def row(**kw):
    defaults = dict(
        name="Test Restaurant", location="Bangalore",
        cuisines="North Indian, Chinese", average_cost_for_two=800,
        aggregate_rating=4.5, votes=500, rest_type="Casual Dining"
    )
    defaults.update(kw)
    return pd.DataFrame([defaults])

passed = failed = 0

def check(desc, cond):
    global passed, failed
    if cond:
        print(f"  ✅  {desc}")
        passed += 1
    else:
        print(f"  ❌  {desc}")
        failed += 1

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 1: Basic preprocessing pipeline ──")
r = PP().process(row())
check("1 restaurant returned",         len(r) == 1)
check("location title-cased",          r[0].location == "Bangalore")
check("cuisines split to list",        r[0].cuisines == ["north indian", "chinese"])
check("budget_tier=medium for ₹800",   r[0].budget_tier == "medium")
check("rating=4.5 preserved",          r[0].rating == 4.5)
check("id is non-empty string",        bool(r[0].id))

print("\n── Test 2: Cuisine edge cases (EC-P-01, EC-P-02, EC-P-07) ──")
check("null cuisine → []",             PP().process(row(cuisines=None))[0].cuisines == [])
check("int 0 cuisine → []",            PP().process(row(cuisines=0))[0].cuisines == [])
check("False cuisine → []",            PP().process(row(cuisines=False))[0].cuisines == [])
check("cuisines are lowercase",        all(c == c.lower() for c in PP().process(row())[0].cuisines))
r12 = PP().process(row(cuisines=",".join([f"Cuisine{i}" for i in range(12)])))
check("12 cuisines all preserved",     len(r12[0].cuisines) == 12)

print("\n── Test 3: Rating coercion (EC-P-03) ──")
check("'NEW' → 0.0",                   PP().process(row(aggregate_rating="NEW"))[0].rating == 0.0)
check("'-' → 0.0",                     PP().process(row(aggregate_rating="-"))[0].rating == 0.0)
check("6.0 clamped to 5.0",           PP().process(row(aggregate_rating=6.0))[0].rating == 5.0)
check("-1 clamped to 0.0",            PP().process(row(aggregate_rating=-1))[0].rating == 0.0)
check("4.5 preserved as-is",          PP().process(row(aggregate_rating=4.5))[0].rating == 4.5)

print("\n── Test 4: Cost coercion (EC-P-04) ──")
check("cost=0 → 0",                    PP().process(row(average_cost_for_two=0))[0].cost_for_two == 0)
check("cost=-500 → 0",                 PP().process(row(average_cost_for_two=-500))[0].cost_for_two == 0)
check("cost=None → 0",                 PP().process(row(average_cost_for_two=None))[0].cost_for_two == 0)

print("\n── Test 5: Location normalisation (EC-P-06, EC-P-09) ──")
check("lowercase → title-case",       PP().process(row(location="bangalore"))[0].location == "Bangalore")
check("strips whitespace",             PP().process(row(location="  Bangalore  "))[0].location == "Bangalore")
check("Bengaluru → Bangalore",         PP().process(row(location="Bengaluru"))[0].location == "Bangalore")
check("Bombay → Mumbai",               PP().process(row(location="Bombay"))[0].location == "Mumbai")
check("DELHI → Delhi",                 PP().process(row(location="DELHI"))[0].location == "Delhi")
check("Calcutta → Kolkata",            PP().process(row(location="Calcutta"))[0].location == "Kolkata")

print("\n── Test 6: Null/empty row dropping (EC-P-08) ──")
check("null name → dropped",           len(PP().process(row(name=None))) == 0)
check("empty name → dropped",          len(PP().process(row(name=""))) == 0)
check("null location → dropped",       len(PP().process(row(location=None))) == 0)
check("rating=0 NOT dropped",          len(PP().process(row(aggregate_rating="NEW"))) == 1)

print("\n── Test 7: Deduplication (EC-P-05) ──")
dup = pd.concat([row(votes=100), row(votes=999)], ignore_index=True)
r_dup = PP().process(dup)
check("exact duplicates → 1 kept",     len(r_dup) == 1)
check("higher-vote entry kept",        r_dup[0].votes == 999)

print("\n── Test 8: Budget tier boundary values (EC-P-10) ──")
for cost, expected in [(0,"low"),(500,"low"),(501,"medium"),(1500,"medium"),(1501,"high"),(5000,"high")]:
    r_b = PP().process(row(average_cost_for_two=cost))
    check(f"₹{cost} → '{expected}'",   r_b[0].budget_tier == expected)

print("\n── Test 9: SchemaError on missing column (EC-D-04) ──")
try:
    PP().process(row().drop(columns=["aggregate_rating"]))
    check("SchemaError raised", False)
except (ValueError, SchemaError) as e:
    check("SchemaError with column name", "aggregate_rating" in str(e))

print("\n── Test 10: Restaurant model helpers ──")
rest = Restaurant("1","Spice Garden","Delhi",["north indian","chinese"],800,4.5,1200,"Casual Dining","medium")
check("cuisine_display formats correctly",  "North Indian" in rest.cuisine_display())
check("rating_display with 4.5",           "4.5" in rest.rating_display())
check("cost_display shows ₹",             "₹" in rest.cost_display())
check("rating 0.0 → 'Not yet rated'",     Restaurant("2","X","Y",rating=0.0).rating_display() == "Not yet rated")
check("cost 0 → 'Cost not available'",    Restaurant("2","X","Y",cost_for_two=0).cost_display() == "Cost not available")
check("matches_cuisine partial=True",      rest.matches_cuisine("indian", partial=True))
check("matches_cuisine partial=False",     not rest.matches_cuisine("indian", partial=False))
check("to_prompt_dict has required keys",  all(k in rest.to_prompt_dict() for k in ["id","name","cuisines","rating"]))

print("\n── Test 11: RestaurantRepository ──")
restaurants = PP().process(pd.DataFrame([
    dict(name="A",location="Delhi",   cuisines="North Indian", average_cost_for_two=500,  aggregate_rating=4.0, votes=100, rest_type=""),
    dict(name="B",location="Delhi",   cuisines="Chinese",      average_cost_for_two=1000, aggregate_rating=3.5, votes=50,  rest_type=""),
    dict(name="C",location="Mumbai",  cuisines="Italian",      average_cost_for_two=2000, aggregate_rating=4.8, votes=300, rest_type=""),
]))
repo = RestaurantRepository(restaurants)
check("count() = 3",                       repo.count() == 3)
check("get_all() returns all 3",           len(repo.get_all()) == 3)
check("get_locations() sorted",           repo.get_locations() == sorted(repo.get_locations()))
check("'Delhi' in get_locations()",        "Delhi" in repo.get_locations())
check("find_by_location Delhi → 2",        len(repo.find_by_location("Delhi")) == 2)
check("find_by_location case-insensitive", len(repo.find_by_location("delhi")) == 2)
check("location_exists('Mumbai')",         repo.location_exists("Mumbai"))
check("location_exists('Chennai') = False",not repo.location_exists("Chennai"))
check("get_cuisines() has 'North Indian'", "North Indian" in repo.get_cuisines())
check("get_by_id returns correct obj",    repo.get_by_id(restaurants[0].id) is restaurants[0])
check("get_by_id unknown → None",         repo.get_by_id("999999") is None)
check("suggest_locations substring match",len(repo.suggest_locations("Del")) > 0)

print(f"\n{'='*52}")
print(f"  Results:  {passed} passed  /  {failed} failed")
if failed == 0:
    print("  🎉  ALL TESTS PASSED — Phase 1 fully verified!")
else:
    print("  ⚠️   Some tests FAILED — review output above.")
print(f"{'='*52}\n")
