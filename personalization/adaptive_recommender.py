import pandas as pd

def load_user(user_id, users_csv="data/users_draft1.csv"):
    df = pd.read_csv(users_csv)
    user = df[df["user_id"] == user_id]
    if user.empty:
        raise ValueError(f"No user found with user_id={user_id}")
    return user.iloc[0]

def load_pois(pois_csv="data/POIs_draft1.csv"):
    return pd.read_csv(pois_csv)

def load_user_history(user_id, history_csv="data/user_history_draft1.csv"):
    df = pd.read_csv(history_csv)
    return df[df["user_id"] == user_id]

def recommend_pois(user_id, top_n=5):
    user = load_user(user_id)
    pois = load_pois()
    history = load_user_history(user_id)

    # Basic personalization filters
    preferred_climate = user["preferred_climate"]
    preferred_budget = user["preferred_budget"]
    interests = [cat.strip() for cat in user["interest_categories"].split(",")]

    # Filter POIs by climate and budget matching user preferences
    filtered = pois[
        (pois["climate"].str.lower() == preferred_climate.lower()) &
        (pois["budget"].str.lower() == preferred_budget.lower()) &
        (pois["category"].str.lower().isin([i.lower() for i in interests]))
    ].copy()

    if filtered.empty:
        print("No POIs match your preferences exactly. Relaxing filters...")
        # Relax the filter by only budget and interest category
        filtered = pois[
            (pois["budget"].str.lower() == preferred_budget.lower()) &
            (pois["category"].str.lower().isin([i.lower() for i in interests]))
        ].copy()

    # Score POIs by rating and user history (liked or booked increases score)
    filtered["score"] = filtered["rating"] / 5.0  # normalize rating between 0-1

    liked_pois = history[history["liked"] == 1]["poi_id"].tolist()
    filtered["final_score"] = filtered.apply(
        lambda row: row["score"] + (0.5 if row["poi_id"] in liked_pois else 0), axis=1
    )

    recommendations = filtered.sort_values("final_score", ascending=False).head(top_n)

    return recommendations[["poi_id", "name", "category", "rating", "final_score"]]

def main():
    user_id = 2  #testing
    print(f"Top POI recommendations for user {user_id}:")

    try:
        recommendations = recommend_pois(user_id)
        print(recommendations.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
