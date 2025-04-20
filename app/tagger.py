import json, os
from ollama import Client  # pip install ollama-python
# Fallback: import openai

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

SYSTEM_PROMPT = """You are a professional music curator..."""

def tag_track(features: dict) -> dict:
    """Call local LLM; return tags as JSON."""
    prompt = f"{SYSTEM_PROMPT}\n\nFEATURES:\n{json.dumps(features, indent=2)}"
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    # response['response'] is the raw string; parse as JSON
    return json.loads(response["response"])
