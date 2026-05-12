"""
Multi-Agent Travel Intelligence System - Streamlit UI
Mercedes-Benz GenAI Internship Round 2
Author: Harsha Bodhe
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import os

st.set_page_config(
    page_title="Travel Intelligence System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stButton>button {
    background: linear-gradient(135deg, #0f3460, #533483);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e, #0f3460); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;">
    <h1 style="color: white; margin: 0;">✈️ Multi-Agent Travel Intelligence System</h1>
    <p style="color: #a0aec0;">Powered by LangGraph · Groq LLM · 4 Specialised Agents</p>
    <p style="color: #718096; font-size: 0.9rem;">Mercedes-Benz GenAI Internship Round 2 | Harsha Bodhe</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Check if keys already loaded from .env
    env_groq = os.getenv("GROQ_API_KEY", "")
    env_weather = os.getenv("OPENWEATHER_API_KEY", "")

    if env_groq:
        st.success("✅ Groq API Key loaded from .env")
        groq_api_key = env_groq
    else:
        groq_api_key = st.text_input(
            "Groq API Key *",
            type="password",
            placeholder="gsk_...",
            help="Get free key at console.groq.com"
        )

    if env_weather:
        st.success("✅ OpenWeather Key loaded from .env")
        openweather_key = env_weather
    else:
        openweather_key = st.text_input(
            "OpenWeather API Key (Optional)",
            type="password",
            placeholder="For live weather data",
            help="Get free key at openweathermap.org"
        )

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if openweather_key:
        os.environ["OPENWEATHER_API_KEY"] = openweather_key

    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("**1. 🧠 Orchestrator Agent**")
    st.caption("Decomposes travel query into structured sub-tasks")
    st.markdown("**2. 🌤️ Weather Agent**")
    st.caption("Fetches real-time weather via OpenWeatherMap API")
    st.markdown("**3. 💰 Budget Agent**")
    st.caption("Estimates flights, hotels, food and activities")
    st.markdown("**4. 📋 Aggregator Agent**")
    st.caption("Synthesizes all data into final travel plan")
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- LangGraph (multi-agent orchestration)\n- Groq llama-3.3-70b\n- OpenWeatherMap API\n- Streamlit UI\n- Python")

st.markdown("### 💬 Ask Your Travel Question")
st.markdown("**Try these examples:**")

examples = [
    "I am traveling from Frankfurt to Tokyo for 5 days. What is the weather and budget?",
    "What is the minimum budget to travel from Germany to Iceland for 10 days?",
    "Planning a trip from Berlin to Barcelona for 7 days. Weather and budget please.",
    "I want to visit Thailand from Munich for 2 weeks on a budget."
]

selected_example = None
c1, c2 = st.columns(2)
for i, example in enumerate(examples):
    col = c1 if i % 2 == 0 else c2
    with col:
        if st.button(f"📍 {example[:50]}...", key=f"ex_{i}"):
            selected_example = example

st.markdown("---")
query = st.text_area(
    "Your Travel Query:",
    value=selected_example if selected_example else "",
    placeholder="e.g. I am traveling from Frankfurt to Tokyo for 5 days...",
    height=100
)

if st.button("🚀 Get Travel Intelligence", use_container_width=True):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar. Get free key at console.groq.com")
    elif not query.strip():
        st.warning("Please enter a travel query first.")
    else:
        try:
            from agents import run_travel_intelligence
        except ImportError as e:
            st.error(f"Import error: {e}")
            st.stop()

        st.markdown("---")
        st.markdown("### 🔄 Agent Workflow — Live Status")
        cols = st.columns(4)
        labels = ["🧠 Orchestrator", "🌤️ Weather", "💰 Budget", "📋 Aggregator"]
        placeholders = []
        for col, label in zip(cols, labels):
            with col:
                p = st.empty()
                p.info(f"{label}\n\n⏳ Waiting")
                placeholders.append(p)

        with st.spinner("Agents working on your travel query..."):
            placeholders[0].warning(f"{labels[0]}\n\n🔄 Running")
            try:
                result = run_travel_intelligence(query)

                for p, label in zip(placeholders, labels):
                    p.success(f"{label}\n\n✅ Done")

                st.markdown("---")
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Travel Plan",
                    "🌤️ Weather",
                    "💰 Budget",
                    "🔍 Agent Data"
                ])

                with tab1:
                    st.markdown("### Your Complete Travel Plan")
                    st.markdown(result.get("final_response", "No response generated"))

                with tab2:
                    st.markdown("### Weather Intelligence")
                    weather = result.get("weather_data", {})
                    if weather and not weather.get("skipped") and "temperature_celsius" in weather:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("🌡️ Temperature", f"{weather.get('temperature_celsius')}°C")
                        m2.metric("💧 Humidity", f"{weather.get('humidity')}%")
                        m3.metric("💨 Wind Speed", f"{weather.get('wind_speed_ms')} m/s")
                        st.info(f"**Conditions:** {weather.get('description', 'N/A')}")
                        if weather.get("travel_tip"):
                            st.success(f"**Packing Tip:** {weather.get('travel_tip')}")
                        st.caption(f"Source: {weather.get('source', 'Unknown')}")
                    else:
                        st.json(weather)

                with tab3:
                    st.markdown("### Budget Intelligence")
                    budget = result.get("budget_data", {})
                    if budget and "costs" in budget:
                        costs = budget["costs"]
                        m1, m2, m3 = st.columns(3)
                        m1.metric("✈️ Flights", f"€{costs.get('flights_roundtrip_eur', 'N/A')}")
                        m2.metric("🏨 Hotel Total", f"€{costs.get('accommodation_total_eur', 'N/A')}")
                        m3.metric("🍽️ Food Total", f"€{costs.get('food_total_eur', 'N/A')}")
                        m4, m5 = st.columns(2)
                        m4.metric("💰 Minimum Budget", f"€{budget.get('total_minimum_eur', 'N/A')}")
                        m5.metric("💎 Recommended Budget", f"€{budget.get('total_recommended_eur', 'N/A')}")
                        tips = budget.get("money_saving_tips", [])
                        if tips:
                            st.markdown("**💡 Money Saving Tips:**")
                            for tip in tips:
                                st.markdown(f"- {tip}")
                        platforms = budget.get("best_booking_platforms", [])
                        if platforms:
                            st.markdown(f"**🔗 Best Platforms:** {', '.join(platforms)}")
                    else:
                        st.json(budget)

                with tab4:
                    st.markdown("### Agent Outputs — Technical View")
                    st.markdown("**Agent 1 — Query Decomposition:**")
                    st.json(result.get("decomposed_tasks", {}))
                    st.markdown("**Agent 2 — Weather Data:**")
                    st.json(result.get("weather_data", {}))
                    st.markdown("**Agent 3 — Budget Data:**")
                    st.json(result.get("budget_data", {}))

            except Exception as e:
                for p, label in zip(placeholders, labels):
                    p.error(f"{label}\n\n❌ Error")
                st.error(f"Error: {str(e)}")
                st.exception(e)

st.markdown("---")
st.caption("Multi-Agent Travel Intelligence System | LangGraph + Groq | Mercedes-Benz GenAI Internship Round 2 | Harsha Bodhe")