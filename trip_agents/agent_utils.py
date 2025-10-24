import pandas as pd
from ml_models.destination_classifier import predict_category
from ml_models.recommender import hybrid_recommend
from ml_models.optimizer import optimize_itinerary

def classify_user_destination(preferences: dict) -> str:
    """
    Use ML model to predict a suitable travel category based on user preferences.
    """
    return predict_category(
        climate=preferences["climate"],
        location=preferences["location"],
        budget=preferences["budget"]
    )

def get_ranked_pois(user_id: int) -> pd.DataFrame:
    df = hybrid_recommend(user_id=user_id)
    print(f"Top POIs for user {user_id}:")
    print(df[["poi_id", "name", "score"]].head())
    return df


def generate_itinerary(pois_df: pd.DataFrame, trip_days: int = 3, budget: str = "medium") -> pd.DataFrame:
    """
    Generate a structured itinerary, enforcing geographic feasibility and diversity.
    Filters POIs to a single country and city if necessary.
    """
    if "country" in pois_df.columns and pois_df["country"].nunique() > 1:
        best_country = pois_df.groupby("country")["score"].mean().idxmax()
        print(f"Filtering POIs to single country: {best_country}")
        pois_df = pois_df[pois_df["country"] == best_country]

    if "location" in pois_df.columns and pois_df["location"].nunique() > 1:
        best_city = pois_df.groupby("location")["score"].mean().idxmax()
        print(f"Filtering POIs to single city: {best_city}")
        pois_df = pois_df[pois_df["location"] == best_city]

    try:
        return optimize_itinerary(
            recommended_pois=pois_df,
            trip_days=trip_days,
            max_budget=budget,
        )
    except ValueError as e:
        print(f"Error during itinerary generation: {e}")
        return pd.DataFrame()

