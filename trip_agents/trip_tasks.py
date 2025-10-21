from crewai import Task
from trip_agents.trip_agents import planner_agent, scoring_agent, optimizer_agent

planner_task = Task(
    description="Analyze user preferences to determine the best travel category.",
    expected_output="Recommended category: e.g., museums, hiking, beaches.",
    agent=planner_agent
)

scoring_task = Task(
    description="Use user_id to score POIs based on preferences and rank the top ones.",
    expected_output="Top POIs with scores.",
    agent=scoring_agent
)

optimizer_task = Task(
    description=(
        "Use top POIs to create a 3-day itinerary. Ensure the plan fits within the user's budget, "
        "includes a diverse set of activities per day, and stays geographically feasible. "
        "Avoid recommending POIs from multiple distant countries or cities in a short trip. "
        "Unless specified otherwise, assume the trip should be in a single city or country region."
    ),
    expected_output="Optimized itinerary with day/slot structure.",
    agent=optimizer_agent
)
