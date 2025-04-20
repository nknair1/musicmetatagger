# app/tagger.py
import json
import os
from ollama import Client
import argparse
from pathlib import Path
from app.analyzer import extract_features

# --- Configuration ---
# Use environment variables for flexibility, falling back to defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

try:
    client = Client(host=OLLAMA_HOST)
except Exception as e:
    print(f"Error connecting to Ollama at {OLLAMA_HOST}: {e}")
    print("Make sure the Ollama server is running.")
    client = None # Set client to None if connection fails

# --- Prompt Engineering ---
SYSTEM_PROMPT = """You are an expert music analyst and tagger. Given a set of audio features extracted from a music track, your task is to generate relevant semantic tags. Analyze the provided features (tempo, key, MFCC patterns indicative of timbre) to infer the following:

- primary_genre: The single most likely main genre (e.g., Rock, Pop, Electronic, Jazz, Classical, Hip-Hop, Folk, Blues).
- sub_genres: A list of 1-3 specific sub-genres (e.g., Synthwave, Indie Pop, Deep House, Bebop, Baroque, Boom Bap, Singer-Songwriter).
- mood_tags: A list of 2-5 descriptive mood keywords (e.g., Energetic, Melancholic, Relaxing, Uplifting, Intense, Dreamy, Nostalgic).
- instruments_likely: A list of 2-4 instruments likely present based on timbre analysis (e.g., Synthesizer, Electric Guitar, Acoustic Drum Kit, Piano, Vocals, Bass Guitar, Strings).

Return ONLY a valid JSON object containing these keys, like this example:
{
  "primary_genre": "Electronic",
  "sub_genres": ["Synthwave", "Chillwave"],
  "mood_tags": ["Nostalgic", "Dreamy", "Relaxing"],
  "instruments_likely": ["Synthesizer", "Drum Machine", "Bass Guitar"]
}"""

# --- Tagging Function ---
def tag_track(features: dict) -> dict | None:
    """
    Takes audio features, sends them to the LLM via Ollama, and returns parsed tags.
    Returns None if Ollama client isn't available or if parsing fails.
    """
    if client is None:
        print("Ollama client not available. Cannot tag track.")
        return None

    # Construct the user part of the prompt using the features
    user_prompt = f"Analyze the following audio features and generate the semantic tags:\n\nFEATURES:\n{json.dumps(features, indent=2)}"

    try:
        print(f"Sending request to Ollama model: {OLLAMA_MODEL}...") # Debug print
        response = client.generate(
            model=OLLAMA_MODEL,
            system=SYSTEM_PROMPT, # Use the dedicated system parameter
            prompt=user_prompt,
            format="json" # Instruct Ollama to ensure the output is JSON
        )
        print("Received response from Ollama.") # Debug print

        # The response content is in response['response']
        response_text = response.get("response", "")
        print(f"Raw response text:\n{response_text}") # Debug print

        # Attempt to parse the JSON string from the response
        tags = json.loads(response_text)
        return tags

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response from Ollama: {e}")
        print(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        print(f"Error during Ollama API call: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate semantic tags for a music audio file.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., test.wav)")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)

    if not audio_path.is_file():
        print(f"Error: Audio file not found at {audio_path}")
    else:
        print(f"1. Extracting features from: {audio_path}...")
        features = extract_features(audio_path)

        if features:
            print("2. Features extracted successfully. Attempting to tag...")
            generated_tags = tag_track(features)

            if generated_tags:
                print("\n3. Generated Tags:")
                print(json.dumps(generated_tags, indent=2))
            else:
                print("\n3. Failed to generate tags.")
        else:
            print("Failed to extract features.")