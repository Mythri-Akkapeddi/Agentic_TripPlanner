from ml_models.preference_scorer import train_preference_model, score_pois_for_user

train_preference_model()

user_id = 1
recommendations = score_pois_for_user(user_id)

print(recommendations[["poi_id", "name", "score"]].head())
