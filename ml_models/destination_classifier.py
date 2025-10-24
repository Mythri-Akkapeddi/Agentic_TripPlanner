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
    # Strip whitespace in all string columns
    for col in ["climate", "location", "budget", "category"]:
        data[col] = data[col].astype(str).str.strip()


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
    print("\nTrained encoder classes:")
    for col, le in encoders.items():
        print(f"{col}: {list(le.classes_)}")


def predict_category(climate, location, budget):
    from sklearn.exceptions import NotFittedError
    import numpy as np

    model = load(MODEL_PATH)
    encoders = load(ENCODER_PATH)

    input_df = pd.DataFrame([{
        "climate": climate,
        "location": location,
        "budget": budget
    }])

    for col in input_df.columns:
        le = encoders[col]
        value = input_df[col].iloc[0]

        if value not in le.classes_:
            print(f"Unseen label '{value}' for column '{col}'. Using fallback: most frequent class.")
            # Use most frequent class as fallback
            most_common = le.classes_[0]
            input_df[col] = le.transform([most_common])
        else:
            input_df[col] = le.transform([value])

    try:
        prediction = model.predict(input_df)[0]
    except NotFittedError:
        raise ValueError("Model has not been trained.")

    category = encoders["category"].inverse_transform([prediction])[0]
    return category

