from ml_models.destination_classifier import predict_category
from ml_models.recommender import hybrid_recommend
from ml_models.optimizer import optimize_itinerary

def classify_user_destination(preferences: dict) -> str:
    return predict_category(
        climate=preferences["climate"],
        location=preferences["location"],
        budget=preferences["budget"]
    )

def get_ranked_pois(user_id: int) -> list:
    return hybrid_recommend(user_id=user_id)

def generate_itinerary(pois_df, trip_days=3, budget="medium"):
    return optimize_itinerary(
        recommended_pois=pois_df,
        trip_days=trip_days,
        max_budget=budget,
    )
