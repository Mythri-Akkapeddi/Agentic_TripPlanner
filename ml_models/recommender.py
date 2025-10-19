import pandas as pd
from ml_models.preference_scorer import score_pois_for_user

def hybrid_recommend(
    user_id: int,
    poi_file: str = "data/POIs_draft1.csv",
    history_file: str = "data/user_history_draft1.csv",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Combines content-based scoring (Naive Bayes) with future potential 
    for collaborative filtering to recommend POIs for a given user.
    """
    # Preference scores
    scored_pois = score_pois_for_user(user_id, poi_file, history_file)

    # [Optional] Add collaborative filtering / similarity signals later

    # Return top N POIs
    return scored_pois[["poi_id", "name", "score"]].head(top_n)
