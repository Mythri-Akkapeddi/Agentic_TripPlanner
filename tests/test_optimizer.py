from ml_models.recommender import hybrid_recommend
from ml_models.optimizer import optimize_itinerary

def test_itinerary_optimizer():
    user_id = 1
    trip_days = 3
    slots_per_day = 2
    max_budget = "medium"

    print(f"\nTesting itinerary for User {user_id} | {trip_days} days | Budget: {max_budget}")

    recommendations = hybrid_recommend(user_id, top_n=20)

    print("\nTop POIs before optimization:")
    print(recommendations.head(5))

    if recommendations["location"].nunique() > 1:
        top_city = recommendations["location"].value_counts().idxmax()
        print(f"\nFiltering POIs to top city: {top_city}")
        recommendations = recommendations[recommendations["location"] == top_city]

    itinerary = optimize_itinerary(
        recommended_pois=recommendations,
        trip_days=trip_days,
        slots_per_day=slots_per_day,
        max_budget=max_budget
    )

    print("\nFinal Optimized Itinerary:")
    print(itinerary)

    assert not itinerary.empty, "Itinerary is empty!"
    assert len(itinerary) <= trip_days * slots_per_day, "Too many POIs in itinerary!"
    assert "day" in itinerary.columns and "time_of_day" in itinerary.columns, "Missing itinerary structure!"

if __name__ == "__main__":
    test_itinerary_optimizer()
