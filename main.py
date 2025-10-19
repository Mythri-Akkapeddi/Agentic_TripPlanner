from crewai import Crew, Process
from trip_agents.trip_agents import planner_agent, scoring_agent, optimizer_agent
from trip_agents.trip_tasks import planner_task, scoring_task, optimizer_task
from trip_agents.agent_utils import (
    classify_user_destination,
    get_ranked_pois,
    generate_itinerary
)

def main(user_id: int):
    print(f"Running Trip Planning Crew for User {user_id}...")

    crew = Crew(
        agents=[planner_agent, scoring_agent, optimizer_agent],
        tasks=[planner_task, scoring_task, optimizer_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff(inputs={"user_id": user_id})

    print("\nFinal Output:\n")
    print(result)

if __name__ == "__main__":
    main(user_id=1)
