import pandas as pd
import os

USERS_FILE = "data/users_draft1.csv"

class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.name = None
        self.preferred_climate = None
        self.preferred_budget = None
        self.interest_categories = []
        self.trip_duration_days = None
        self.load_user_profile()

    def load_user_profile(self):
        if not os.path.exists(USERS_FILE):
            raise FileNotFoundError(f"{USERS_FILE} not found.")

        df = pd.read_csv(USERS_FILE)
        user_data = df[df["user_id"] == self.user_id]

        if user_data.empty:
            raise ValueError(f"User with id {self.user_id} not found.")

        user = user_data.iloc[0]
        self.name = user["name"]
        self.preferred_climate = user["preferred_climate"]
        self.preferred_budget = user["preferred_budget"]
        self.interest_categories = [cat.strip() for cat in user["interest_categories"].split(",")]
        self.trip_duration_days = int(user["trip_duration_days"])

    def __repr__(self):
        return (f"UserProfile(user_id={self.user_id}, name={self.name}, "
                f"preferred_climate={self.preferred_climate}, preferred_budget={self.preferred_budget}, "
                f"interest_categories={self.interest_categories}, trip_duration_days={self.trip_duration_days})")
