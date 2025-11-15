# tools.py
import json
from typing import List, Dict
import os
import requests

# ----------------------------
# NOTE:
# - This module includes a simple mocked web_search to keep demo runnable offline.
# - If you have a SerpAPI key, uncomment and fill real calls as indicated.
# ----------------------------

def web_search_mock(query: str) -> str:
    """
    Return a short mocked summary for a query.
    Use this for offline demos (Kaggle Notebook, Colab without SerpAPI).
    """
    return f"Mocked top result summary for '{query}'. Use a real search API (SerpAPI) to replace this."

def web_search(query: str, use_serpapi: bool = False, serpapi_key: str = None) -> str:
    if use_serpapi and serpapi_key:
        # Example (commented) usage for SerpAPI:
        # resp = requests.get("https://serpapi.com/search", params={
        #     "q": query,
        #     "api_key": serpapi_key,
        #     "engine": "google"
        # })
        # data = resp.json()
        # # Choose an appropriate field from the returned JSON as summary
        # return data.get("organic_results", [{}])[0].get("snippet", "No snippet found")
        pass
    return web_search_mock(query)

# Quiz generator: uses the LLM (agent will call Groq via agent.ask_llm)
def parse_quiz_output_to_list(text: str):
    """
    Try to parse a JSON list returned by the LLM. If parse fails, return fallback.
    """
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback: not ideal but keeps program robust
    return []

def generate_quiz_prompt(topic: str, n: int = 5) -> str:
    return (
        "Generate {n} multiple-choice questions for the topic: '{topic}'. "
        "Return a JSON array. Each item should be an object with keys: "
        "'question' (string), 'options' (array of 4 strings), 'answer_index' (0-3). "
        "Ensure questions are concise and beginner-friendly."
    ).format(n=n, topic=topic)
