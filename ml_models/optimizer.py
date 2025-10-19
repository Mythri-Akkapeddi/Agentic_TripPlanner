import pandas as pd
from typing import List

def optimize_itinerary(
    recommended_pois: pd.DataFrame,
    trip_days: int = 3,
    slots_per_day: int = 2,
    max_budget: str = "medium",
    enforce_diversity: bool = True
) -> pd.DataFrame:
    """
    Optimizes an itinerary from recommended POIs using constraints like:
    - Budget (filters POIs to match user budget)
    - Diversity (no repeating categories in same day)
    - Duration (max POIs = days * slots_per_day)
    
    Parameters:
        recommended_pois (pd.DataFrame): Output from recommender with 'score' column
        trip_days (int): Total days in user's trip
        slots_per_day (int): AM/PM slots per day
        max_budget (str): User's budget preference ("low", "medium", "high")
        enforce_diversity (bool): If True, ensures category diversity
        
    Returns:
        pd.DataFrame: Structured itinerary with 'day', 'time_of_day', 'poi_id', 'name', etc.
    """
    required_columns = ["budget", "category", "duration_hours", "rating", "score"]
    for col in required_columns:
        if col not in recommended_pois.columns:
            raise ValueError(f"Missing required column in input: '{col}'")


    budget_levels = ["low", "medium", "high"]
    if max_budget not in budget_levels:
        raise ValueError(f"Invalid budget level: {max_budget}. Use: {budget_levels}")

    # Filtering POIs within the budget range
    allowed_budgets = budget_levels[:budget_levels.index(max_budget)+1]
    filtered = recommended_pois[recommended_pois["budget"].isin(allowed_budgets)].copy()

    # Sorting by descending score
    filtered = filtered.sort_values(by="score", ascending=False)

    # Building itinerary
    max_pois = trip_days * slots_per_day
    selected_pois = []
    used_per_day = {day: set() for day in range(1, trip_days + 1)}  # for diversity

    i = 0
    for _, row in filtered.iterrows():
        if len(selected_pois) >= max_pois:
            break

        # Calculate which day and slot this POI would go into
        current_index = len(selected_pois)
        day = current_index // slots_per_day + 1
        slot = current_index % slots_per_day
        time_of_day = "AM" if slot == 0 else "PM"

        # Enforce diversity: don't repeat category in same day
        if enforce_diversity:
            if row["category"] in used_per_day[day]:
                continue
            used_per_day[day].add(row["category"])

        # Add day/time info and append
        poi = row.copy()
        poi["day"] = day
        poi["time_of_day"] = time_of_day
        selected_pois.append(poi)

    # Final itinerary dataframe
    itinerary_df = pd.DataFrame(selected_pois)
    itinerary_df = itinerary_df[
        ["day", "time_of_day", "poi_id", "name", "category", "budget", "duration_hours", "rating", "score"]
    ].reset_index(drop=True)

    return itinerary_df
