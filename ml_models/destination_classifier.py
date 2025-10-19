import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import os

MODEL_PATH = "models/destination_classifier.pkl"
ENCODER_PATH = "models/destination_label_encoders.pkl"

def load_data(poi_file):
    return pd.read_csv(poi_file)

def preprocess_data(data):
    encoders = {}
    X = data[["climate", "location", "budget"]].copy()  #True copy to accommodate SettingWithCopyWarning
    y = data["category"]
    
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    encoders["category"] = y_le

    return X, y, encoders


def train_model(poi_file="data/POIs_draft1.csv"):
    data = load_data(poi_file)
    X, y, encoders = preprocess_data(data)

    model = DecisionTreeClassifier()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    dump(model, MODEL_PATH)
    dump(encoders, ENCODER_PATH)

    print("Destination classifier trained and saved.")

def predict_category(climate, location, budget):
    model = load(MODEL_PATH)
    encoders = load(ENCODER_PATH)

    input_df = pd.DataFrame([{
        "climate": climate,
        "location": location,
        "budget": budget
    }])

    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    category = encoders["category"].inverse_transform([prediction])[0]
    return category
