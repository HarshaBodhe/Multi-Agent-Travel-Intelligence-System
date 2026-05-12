# ✈️ Multi-Agent Travel Intelligence System

**Mercedes-Benz GenAI / Agentic AI Engineering Internship — Round 2**
**Author:** Harsha Bodhe | [LinkedIn](https://linkedin.com/in/harsha-bodhe) | [GitHub](https://github.com/HarshaBodhe)

---

## 🎯 Overview

A multi-agent AI system that answers complex travel queries by coordinating 
4 specialised agents using LangGraph orchestration and Groq LLM inference.

**Example queries handled:**
- "I am traveling from Frankfurt to Tokyo for 5 days. What is the weather and budget?"
- "What is the minimum budget to travel from Germany to Iceland for 10 days?"

---

## 🤖 Agent Architecture

| Agent | Role |
|---|---|
| 🧠 Orchestrator Agent | Decomposes travel query into structured sub-tasks |
| 🌤️ Weather Agent | Fetches live weather via OpenWeatherMap API |
| 💰 Budget Agent | Estimates flights, hotels, food and activities |
| 📋 Aggregator Agent | Synthesizes all data into final travel plan |

---

## 🛠️ Tech Stack

- **LangGraph** — Multi-agent orchestration
- **Groq LLM** — llama-3.3-70b-versatile (free)
- **OpenWeatherMap API** — Live weather data (free)
- **Streamlit** — Interactive web UI
- **Python** — Core language

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API keys to .env file
GROQ_API_KEY=your_groq_key
OPENWEATHER_API_KEY=your_weather_key

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
── agents.py        # All 4 agent definitions + LangGraph workflow
├── app.py           # Streamlit frontend UI
├── requirements.txt # Python dependencies
└── .env.example     # Environment variables template
---

*Built for Mercedes-Benz GenAI / Agentic AI Engineering Internship — Round 2*
