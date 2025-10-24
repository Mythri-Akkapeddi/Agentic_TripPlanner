from trip_agents.trip_agents import generate_itinerary
import pandas as pd

def main():
    print("Agentic Trip Planner")
    user_id = 1

    location = input("Enter a city you'd like to visit (e.g., Paris): ").strip()
    country = input("Enter the country: ").strip()
    budget = input("Enter your budget level (low/medium/high): ").strip().lower() or "medium"
    climate = input("Preferred climate (warm/cold): ").strip().lower() or "warm"

    itinerary = generate_itinerary(
        user_id=user_id,
        location=location,
        country=country,
        budget=budget,
        climate=climate,
        trip_days=3
    )

    print("\nYour Optimized Itinerary")
    for _, row in itinerary.iterrows():
        print(f"Day {row['day']} {row['time_of_day']}: {row['name']} ({row['category']}) — "
              f"Budget: {row['budget']} | Rating: {row['rating']}")
        print(f"→ {row['explanation']}\n")

if __name__ == "__main__":
    main()
