from crewai.tools import BaseTool
from typing import List, Dict
import pandas as pd
from trip_agents.agent_utils import (
    classify_user_destination,
    get_ranked_pois,
    generate_itinerary
)

class DestinationClassifierTool(BaseTool):
    name: str = "DestinationClassifier"
    description: str = (
        "Classifies travel category from user preferences. "
        "Valid climates: warm, cold, temperate. "
        "Valid locations: urban, rural, coastal. "
        "Valid budgets: low, medium, high."
    )

    def _run(self, climate: str, location: str, budget: str) -> str:
        user_preferences = {
            "climate": climate,
            "location": location,
            "budget": budget
        }
        return classify_user_destination(user_preferences)


class POIScorerTool(BaseTool):
    name: str = "POIScorer"
    description: str = "Scores and recommends POIs for a given user ID based on preferences."

    def _run(self, user_id: int) -> List[Dict]:
        df = get_ranked_pois(user_id)

        required_cols = [
            "poi_id", "name", "category", "location", "country",
            "climate", "budget", "duration_hours", "rating", "score"
        ]

        for col in required_cols:
            if col not in df.columns:
                if col == "duration_hours":
                    df[col] = 2  # assume avg 2 hours per POI
                elif col == "score" and "rating" in df.columns:
                    df[col] = df["rating"]
                else:
                    df[col] = None

        df = df[required_cols]
        print(f"POIScorerTool returning {len(df)} POIs with columns: {list(df.columns)}")

        return df.to_dict(orient="records")


class ItineraryOptimizerTool(BaseTool):
    name: str = "ItineraryOptimizer"
    description: str = (
        "Creates a 3-day itinerary from POIs while enforcing budget "
        "and geographic feasibility."
    )

    def _run(self, pois_df: List[Dict], trip_days: int = 3, budget: str = "medium") -> str:
        if not pois_df:
            return "No POIs provided to optimizer."

        pois_df = pd.DataFrame(pois_df)
        print("Columns in POIs passed to optimizer:", list(pois_df.columns))

        if "score" not in pois_df.columns:
            print("Missing 'score' column — using 'rating' as fallback.")
            pois_df["score"] = pois_df["rating"] if "rating" in pois_df.columns else 0

        if "duration_hours" not in pois_df.columns:
            print("Missing 'duration_hours' — assigning default value (2 hours).")
            pois_df["duration_hours"] = 2

        pois_df["duration_hours"].fillna(2, inplace=True)
        pois_df["score"].fillna(pois_df.get("rating", 0), inplace=True)

        try:
            itinerary = generate_itinerary(pois_df, trip_days=trip_days, budget=budget)
        except Exception as e:
            print(f"Error in ItineraryOptimizerTool: {e}")
            return f"Error while generating itinerary: {e}"

        if isinstance(itinerary, pd.DataFrame) and not itinerary.empty:
            itinerary_str = itinerary.to_string(index=False)
            print("ItineraryOptimizerTool generated structured itinerary:")
            print(itinerary_str)
            return itinerary_str

        elif isinstance(itinerary, str):
            print("ItineraryOptimizerTool returned string itinerary.")
            return itinerary

        return "Could not generate a valid itinerary with the given constraints."
