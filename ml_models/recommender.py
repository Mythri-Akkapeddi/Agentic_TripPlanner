import pandas as pd
from ml_models.preference_scorer import score_pois_for_user

def hybrid_recommend(
    user_id: int,
    poi_file: str = "data/POIs_draft1.csv",
    history_file: str = "data/user_history_draft1.csv",
    top_n: int = 10
) -> pd.DataFrame:
    # Score POIs
    scored_pois = score_pois_for_user(user_id, poi_file, history_file)
    if scored_pois.empty:
        raise ValueError("No POIs could be scored for this user.")

    # Determine user's base country from liked POIs
    user_history = pd.read_csv(history_file)
    pois = pd.read_csv(poi_file)
    merged = user_history.merge(pois, on="poi_id")
    user_likes = merged[(merged["user_id"] == user_id) & (merged["liked"] == 1)]

    if not user_likes.empty:
        base_country = user_likes["country"].value_counts().idxmax()
        country_filtered = scored_pois[scored_pois["country"] == base_country]
        if len(country_filtered) >= top_n:
            return country_filtered.head(top_n)

    # Fallback: highest average scoring country
    country_avg_scores = scored_pois.groupby("country")["score"].mean().sort_values(ascending=False)
    for country in country_avg_scores.index:
        country_filtered = scored_pois[scored_pois["country"] == country]
        if len(country_filtered) >= top_n:
            return country_filtered.head(top_n)

    # Final fallback: top_n overall
    print("⚠️ Returning top_n POIs without country filter (insufficient regional density).")
    return scored_pois.head(top_n)
