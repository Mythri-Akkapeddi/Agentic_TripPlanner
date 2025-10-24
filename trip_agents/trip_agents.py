from crewai import Agent, Crew
from trip_agents.llm import llm
from trip_agents.trip_tasks import (
    create_trip_planning_task,
    create_scoring_task,
    create_optimizer_task,
    create_explanation_task
)
from ml_models.recommender import hybrid_recommend
from ml_models.optimizer import optimize_itinerary
from ml_models.destination_classifier import predict_category
import pandas as pd


# Agent Definitions

class PlannerAgent(Agent):
    def plan_trip(self, location, budget, climate):
        category = predict_category(climate, location, budget)
        print(f"[PlannerAgent] Predicted category: {category}")
        return category


class ScoringAgent(Agent):
    def score_pois(self, user_id, location, country):
        pois = pd.read_csv("data/POIs_draft1.csv")
        filtered = pois[(pois["country"] == country) & (pois["location"] == location)]
        if filtered.empty:
            raise ValueError(f"No POIs found in {location}, {country}")
        filtered["score"] = filtered["rating"] / 5.0
        return filtered.sort_values("score", ascending=False)


class OptimizerAgent(Agent):
    def build_itinerary(self, pois, trip_days, slots_per_day, max_budget):
        print("[OptimizerAgent] Building itinerary...")
        itinerary = optimize_itinerary(
            recommended_pois=pois,
            trip_days=trip_days,
            slots_per_day=slots_per_day,
            max_budget=max_budget,
            enforce_diversity=True
        )
        return itinerary


"""class ExplanationAgent(Agent):
    def explain_itinerary(self, itinerary_df):
        explanations = []
        for _, row in itinerary_df.iterrows():
            explanations.append(
                f"{row['name']} was selected because it fits your budget ({row['budget']}), "
                f"has a high rating ({row['rating']}), and matches your interest in {row['category']}."
            )
        itinerary_df["explanation"] = explanations
        return itinerary_df"""

class ExplanationAgent(Agent):
    def explain_itinerary(self, itinerary_df):
        """
        Uses the local LLM (Mistral via Ollama) to generate natural-language
        explanations for each POI in the itinerary.
        """
        explanations = []

        for _, row in itinerary_df.iterrows():
            prompt = f"""
            You are a friendly travel assistant.
            Explain in 1–2 sentences why {row['name']} in {row['location']} was chosen 
            for the itinerary. Mention its category ({row['category']}), 
            rating ({row['rating']} stars), and budget level ({row['budget']}).
            Keep it concise, warm, and natural.
            """

            try:
                # Handle different CrewAI LLM APIs
                if hasattr(self.llm, "run"):
                    response = self.llm.run(prompt)
                elif hasattr(self.llm, "call"):
                    response = self.llm.call(prompt)
                elif hasattr(self.llm, "__call__"):
                    response = self.llm(prompt)
                else:
                    raise AttributeError("No valid text generation method found in LLM.")

                explanation = str(response).strip()

            except Exception as e:
                print(f"LLM failed for {row['name']}: {e}")
                explanation = (
                    f"{row['name']} was selected because it fits your budget "
                    f"({row['budget']}), has a high rating ({row['rating']}), "
                    f"and matches your interest in {row['category']}."
                )

            explanations.append(explanation)

        itinerary_df["explanation"] = explanations
        return itinerary_df



# Crew Orchestration Function

def generate_itinerary(
    user_id: int,
    location: str,
    country: str,
    budget: str = "medium",
    climate: str = "warm",
    trip_days: int = 3
):
    print(f"Generating itinerary for {location}, {country}...")

    planner = PlannerAgent(
        llm=llm,
        role="Trip Planner",
        goal="Plan suitable destinations and categories based on user context",
        backstory="An expert travel planner who knows global destinations and user preferences."
    )

    scorer = ScoringAgent(
        llm=llm,
        role="POI Scorer",
        goal="Score POIs by user interest and preference",
        backstory="An experienced travel data analyst ranking points of interest."
    )

    optimizer = OptimizerAgent(
        llm=llm,
        role="Itinerary Optimizer",
        goal="Optimize itinerary with time and budget constraints",
        backstory="A travel logistics specialist who designs efficient trip schedules."
    )

    explainer = ExplanationAgent(
        llm=llm,
        role="Explanation Agent",
        goal="Provide justifications for itinerary decisions",
        backstory="An AI travel assistant skilled at explaining why each choice was made."
    )

    
    planning_task = create_trip_planning_task(planner, location, budget, climate)
    scoring_task = create_scoring_task(scorer, user_id, location, country)
    optimizer_task = create_optimizer_task(optimizer, trip_days, 2, budget)
    explanation_task = create_explanation_task(explainer)

    crew = Crew(
        agents=[planner, scorer, optimizer, explainer],
        tasks=[planning_task, scoring_task, optimizer_task, explanation_task],
        verbose=True
    )

    # Sequential workflow
    category = planner.plan_trip(location, budget, climate)
    pois = scorer.score_pois(user_id, location, country)
    itinerary = optimizer.build_itinerary(pois, trip_days, 2, budget)
    explained = explainer.explain_itinerary(itinerary)

    if explained.empty:
        raise ValueError("No itinerary generated. Try different parameters or check data.")

    return explained
