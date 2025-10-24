import pandas as pd
import os

HISTORY_FILE = "data/user_history_draft1.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    else:
        return pd.DataFrame(columns=["user_id", "poi_id", "liked", "interaction_type", "timestamp"])

def get_user_history(user_id):
    history = load_history()
    return history[history["user_id"] == user_id]

def get_positive_interactions(user_id):
    history = get_user_history(user_id)
    return history[history["liked"] == 1]

def save_interaction(user_id, poi_id, liked, interaction_type="clicked"):
    history = load_history()
    new_entry = pd.DataFrame([{
        "user_id": user_id,
        "poi_id": poi_id,
        "liked": liked,
        "interaction_type": interaction_type,
        "timestamp": pd.Timestamp.now()
    }])
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)
    print(f"Interaction saved for user {user_id} ({'liked' if liked else 'not liked'}).")
