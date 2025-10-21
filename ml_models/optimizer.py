import pandas as pd

def optimize_itinerary(
    recommended_pois: pd.DataFrame,
    trip_days: int = 3,
    slots_per_day: int = 2,
    max_budget: str = "medium",
    enforce_diversity: bool = True
) -> pd.DataFrame:
    required_cols = ["poi_id","name","category","location","budget","duration_hours","rating","score"]
    for col in required_cols:
        if col not in recommended_pois.columns:
            raise ValueError(f"Missing required column in input: '{col}'")

    # Enforcing single base location
    if recommended_pois["location"].nunique() > 1:
        raise ValueError("Points of Interests are from multiple cities. Optimizer expects a single base city.")

    budget_levels = ["low","medium","high"]
    if max_budget not in budget_levels:
        raise ValueError(f"Invalid budget: {max_budget}. Choose from {budget_levels}")

    allowed = budget_levels[:budget_levels.index(max_budget) + 1]
    filtered = recommended_pois[recommended_pois["budget"].isin(allowed)].copy()
    if filtered.empty:
        print("After budget filtering, no Points of Interests remain.")
        return pd.DataFrame()

    filtered = filtered.sort_values(by="score", ascending=False)
    max_pois = trip_days * slots_per_day
    selected = []
    used_cats = {day:set() for day in range(1, trip_days+1)}

    for _, row in filtered.iterrows():
        if len(selected) >= max_pois:
            break
        idx = len(selected)
        day = idx // slots_per_day + 1
        slot = idx % slots_per_day
        time_of_day = "AM" if slot == 0 else "PM"

        if enforce_diversity and (row["category"] in used_cats[day]):
            continue
        used_cats[day].add(row["category"])

        poi = row.copy()
        poi["day"] = day
        poi["time_of_day"] = time_of_day
        selected.append(poi)

    if not selected:
        print("No Points of Interests selected under current constraints.")
        return pd.DataFrame()

    itinerary_df = pd.DataFrame(selected)
    return itinerary_df[["day","time_of_day","location","poi_id","name","category","budget","duration_hours","rating","score"]].reset_index(drop=True)
