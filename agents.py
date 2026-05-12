"""
Multi-Agent Travel Intelligence System
Mercedes-Benz GenAI / Agentic AI Engineering Internship 
Author: Harsha Bodhe
Framework: LangGraph + Groq LLM
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to .env file or sidebar.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )

class TravelState(TypedDict):
    user_query: str
    decomposed_tasks: dict
    weather_data: dict
    budget_data: dict
    final_response: str
    error: str

def orchestrator_agent(state: TravelState) -> TravelState:
    """Agent 1: Decomposes user travel query into structured sub-tasks."""
    llm = get_llm()
    system_prompt = """You are a travel query decomposition expert.
    Analyze the user travel query and extract structured information.
    Always respond with a valid JSON object only — no other text:
    {
        "origin": "departure city or null",
        "destination": "destination city",
        "duration_days": 7,
        "travel_type": "budget",
        "needs_weather": true,
        "needs_budget": true,
        "needs_flights": false,
        "summary": "brief description of what user wants"
    }"""
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Travel query: {state['user_query']}")
        ])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        decomposed = json.loads(content)
        return {**state, "decomposed_tasks": decomposed}
    except Exception as e:
        return {**state, "decomposed_tasks": {
            "destination": "unknown",
            "needs_weather": True,
            "needs_budget": True,
            "needs_flights": False,
            "summary": state["user_query"],
            "error": str(e)
        }}

def weather_agent(state: TravelState) -> TravelState:
    """Agent 2: Fetches real weather data or falls back to LLM estimate."""
    if not state["decomposed_tasks"].get("needs_weather", True):
        return {**state, "weather_data": {"skipped": True}}

    destination = state["decomposed_tasks"].get("destination", "")
    api_key = os.getenv("OPENWEATHER_API_KEY", "")

    # Try live OpenWeatherMap API first
    if api_key and destination and destination != "unknown":
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {"q": destination, "appid": api_key, "units": "metric"}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {**state, "weather_data": {
                    "destination": destination,
                    "temperature_celsius": round(data["main"]["temp"]),
                    "feels_like": round(data["main"]["feels_like"]),
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"].capitalize(),
                    "wind_speed_ms": data["wind"]["speed"],
                    "source": "OpenWeatherMap API (Live)"
                }}
        except:
            pass

    # Fallback: LLM weather estimate
    llm = get_llm()
    duration = state["decomposed_tasks"].get("duration_days", 7)
    system_prompt = """You are a travel weather expert.
    Respond ONLY with a valid JSON object — no other text:
    {
        "destination": "city name",
        "temperature_celsius": 20,
        "feels_like": 18,
        "humidity": 65,
        "description": "Partly cloudy",
        "wind_speed_ms": 5,
        "travel_tip": "packing and clothing recommendation",
        "source": "AI Weather Estimate"
    }"""
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Destination: {destination}, Duration: {duration} days. Provide typical weather.")
        ])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return {**state, "weather_data": json.loads(content)}
    except Exception as e:
        return {**state, "weather_data": {
            "destination": destination,
            "description": "Weather data temporarily unavailable",
            "error": str(e),
            "source": "Unavailable"
        }}

def budget_agent(state: TravelState) -> TravelState:
    """Agent 3: Estimates full travel budget including flights, hotels, food, activities."""
    if not state["decomposed_tasks"].get("needs_budget", True):
        return {**state, "budget_data": {"skipped": True}}

    llm = get_llm()
    origin = state["decomposed_tasks"].get("origin", "Germany")
    destination = state["decomposed_tasks"].get("destination", "")
    duration = state["decomposed_tasks"].get("duration_days", 7)
    travel_type = state["decomposed_tasks"].get("travel_type", "budget")

    system_prompt = """You are an expert travel budget planner with deep global knowledge.
    Respond ONLY with a valid JSON object — no other text:
    {
        "origin": "city",
        "destination": "city",
        "duration_days": 7,
        "travel_type": "budget",
        "costs": {
            "flights_roundtrip_eur": 400,
            "accommodation_per_night_eur": 50,
            "accommodation_total_eur": 350,
            "food_per_day_eur": 30,
            "food_total_eur": 210,
            "activities_total_eur": 100,
            "transport_local_total_eur": 50,
            "misc_eur": 50
        },
        "total_minimum_eur": 900,
        "total_recommended_eur": 1200,
        "money_saving_tips": ["tip1", "tip2", "tip3"],
        "best_booking_platforms": ["Skyscanner", "Booking.com"],
        "currency_note": "local currency and exchange rate info"
    }"""
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Origin: {origin}, Destination: {destination}, Duration: {duration} days, Travel style: {travel_type}")
        ])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return {**state, "budget_data": json.loads(content)}
    except Exception as e:
        return {**state, "budget_data": {
            "error": str(e),
            "destination": destination,
            "note": "Budget estimation temporarily unavailable"
        }}

def aggregator_agent(state: TravelState) -> TravelState:
    """Agent 4: Synthesizes all agent outputs into a final structured travel plan."""
    llm = get_llm()
    context = {
        "original_query": state["user_query"],
        "query_analysis": state["decomposed_tasks"],
        "weather_information": state["weather_data"],
        "budget_information": state["budget_data"]
    }
    system_prompt = """You are a friendly expert travel advisor.
    Using the provided travel intelligence data, create a comprehensive travel plan.
    Format your response clearly with these sections:

    🌍 TRIP OVERVIEW
    🌤️ WEATHER AND PACKING TIPS
    💰 BUDGET BREAKDOWN
    ✈️ PRACTICAL TIPS

    Be specific with numbers, helpful with recommendations, and friendly in tone.
    If any data is unavailable, provide general advice from your knowledge."""
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Travel Intelligence Data:\n{json.dumps(context, indent=2)}")
        ])
        return {**state, "final_response": response.content}
    except Exception as e:
        return {**state, "final_response": f"Error generating response: {str(e)}"}

def build_travel_graph():
    """Builds and compiles the 4-agent LangGraph workflow."""
    workflow = StateGraph(TravelState)

    # Register all agent nodes
    workflow.add_node("orchestrator_agent", orchestrator_agent)
    workflow.add_node("weather_agent", weather_agent)
    workflow.add_node("budget_agent", budget_agent)
    workflow.add_node("aggregator_agent", aggregator_agent)

    # Define the agent execution pipeline
    workflow.set_entry_point("orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "weather_agent")
    workflow.add_edge("weather_agent", "budget_agent")
    workflow.add_edge("budget_agent", "aggregator_agent")
    workflow.add_edge("aggregator_agent", END)

    return workflow.compile()

def run_travel_intelligence(query: str) -> dict:
    """Main entry point — runs the full multi-agent travel intelligence pipeline."""
    graph = build_travel_graph()
    initial_state = TravelState(
        user_query=query,
        decomposed_tasks={},
        weather_data={},
        budget_data={},
        final_response="",
        error=""
    )
    return graph.invoke(initial_state)