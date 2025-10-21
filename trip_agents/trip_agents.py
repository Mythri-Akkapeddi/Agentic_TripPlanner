from crewai import Agent
from trip_agents.llm import llm  

planner_agent = Agent(
    role="Trip Planner",
    goal="Understand user preferences and recommend travel categories",
    backstory="An expert travel consultant who knows global destinations and how to match them to personal tastes",
    llm=llm,              
    verbose=True
)

scoring_agent = Agent(
    role="POI Recommender",
    goal="Score POIs using machine learning models based on user preferences",
    backstory="A smart system that knows how to match places to people using past data and smart scoring algorithms",
    llm=llm,              
    verbose=True
)

optimizer_agent = Agent(
    role="Itinerary Builder",
    goal="Use the best POIs to craft a perfect multi-day travel plan",
    backstory="An itinerary optimizer that balances variety, time, and budget while ensuring all activities are located in the same region to avoid impractical travel",
    llm=llm,             
    verbose=True
)
