from ollama import Client
import os, json

SYSTEM = "You are an SEO copywriter for a music streaming service..."
client = Client()

def build_metadata(track_title: str, artist: str, tags: dict) -> dict:
    user_msg = json.dumps({"title": track_title, "artist": artist, "tags": tags}, indent=2)
    text = client.generate(model=os.getenv("OLLAMA_MODEL", "llama3:8b"),
                           prompt=f"{SYSTEM}\n\n{user_msg}")["response"]
    return json.loads(text)
