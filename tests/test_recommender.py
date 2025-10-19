from ml_models.recommender import hybrid_recommend

def test_hybrid_recommend():
    user_id = 1
    recommendations = hybrid_recommend(user_id, top_n=5)
    
    assert not recommendations.empty, "No recommendations returned."
    assert "score" in recommendations.columns, "Missing 'score' column in output."

    print("Top POI recommendations for user_id=1:")
    print(recommendations)

if __name__ == "__main__":
    test_hybrid_recommend()
