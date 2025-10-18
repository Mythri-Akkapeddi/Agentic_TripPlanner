import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

MODEL_PATH = "models/naive_bayes_scorer.pkl"
ENCODER_PATH = "models/nb_label_encoders.pkl"

def load_and_prepare_data(poi_file, history_file):
    pois = pd.read_csv(poi_file)
    history = pd.read_csv(history_file)

    # Merging user interactions with POI features
    data = history.merge(pois, on="poi_id")

    return data

def preprocess_data(data):
    # Encoding categorical features
    label_encoders = {}
    categorical_cols = ["climate", "category", "budget", "location", "country"]
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Selecting features
    X = data[["climate", "category", "budget", "duration_hours", "rating"]]
    y = data["liked"]

    return X, y, label_encoders

def train_preference_model(poi_file="data/POIs_draft1.csv", history_file="data/user_history_draft1.csv"):
    data = load_and_prepare_data(poi_file, history_file)
    X, y, encoders = preprocess_data(data)

    model = GaussianNB()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    dump(model, MODEL_PATH)
    dump(encoders, ENCODER_PATH)

    print("Naive Bayes preference model trained and saved.")

def score_pois_for_user(user_id, poi_file="data/POIs_draft1.csv", history_file="data/user_history_draft1.csv"):
    model = load(MODEL_PATH)
    encoders = load(ENCODER_PATH)

    pois = pd.read_csv(poi_file)
    history = pd.read_csv(history_file)

    # POIs the user hasn't interacted with
    interacted = history[history["user_id"] == user_id]["poi_id"]
    unseen_pois = pois[~pois["poi_id"].isin(interacted)]

    # Encoding using saved encoders (handle unseen labels by dropping them)
    for col, le in encoders.items():
        valid_mask = unseen_pois[col].isin(le.classes_)
        unseen_pois = unseen_pois[valid_mask]
        unseen_pois[col] = le.transform(unseen_pois[col])

    features = unseen_pois[["climate", "category", "budget", "duration_hours", "rating"]]
    scores = model.predict_proba(features)[:, 1]  # probability of 'liked' class

    unseen_pois["score"] = scores
    return unseen_pois.sort_values(by="score", ascending=False)
