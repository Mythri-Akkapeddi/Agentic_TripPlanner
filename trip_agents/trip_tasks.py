from crewai import Task

def create_trip_planning_task(agent, location, budget, climate):
    return Task(
        description=f"Plan a trip for {location} with a {budget} budget and {climate} climate.",
        expected_output="A suitable trip category (e.g., culture, hiking, beaches).",
        agent=agent
    )

def create_scoring_task(agent, user_id, location, country):
    return Task(
        description=f"Score POIs in {location}, {country} for user {user_id}.",
        expected_output="A ranked list of POIs with relevance scores.",
        agent=agent
    )

def create_optimizer_task(agent, trip_days, slots_per_day, budget):
    return Task(
        description=f"Optimize itinerary for {trip_days} days and {slots_per_day} activities/day within {budget} budget.",
        expected_output="Ordered itinerary dataframe with day/time slots and POIs.",
        agent=agent
    )

def create_explanation_task(agent):
    return Task(
        description="Explain why each POI was selected in the final itinerary.",
        expected_output="Natural language explanation for each POI.",
        agent=agent
    )
