# app/tagger.py
import json
import os
import argparse
import logging
from pathlib import Path
from ollama import Client

# Assuming analyzer is in the same 'app' directory or PYTHONPATH is set correctly
try:
    from .analyzer import extract_features
except ImportError:
    # Fallback for running the script directly during development
    from analyzer import extract_features


# --- Configuration ---
# Configure basic logging (adjust level or format as needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

# Use environment variables for flexibility, falling back to defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b") # Or use "llama3:latest"

# --- Ollama Client Initialization ---
try:
    client = Client(host=OLLAMA_HOST)
    # Optional: Check connection by listing local models
    client.list()
    logger.info(f"Successfully connected to Ollama at {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Error connecting to Ollama at {OLLAMA_HOST}: {e}", exc_info=True)
    logger.error("Please ensure the Ollama server is running (`ollama serve`)")
    client = None # Set client to None if connection fails

# --- Prompt Engineering ---
SYSTEM_PROMPT = """You are an expert music analyst and tagger. Given a set of audio features extracted from a music track, your task is to generate relevant semantic tags. Analyze the provided features:
- tempo_bpm: Beats per minute, indicating speed (e.g., 60=slow, 120=moderate, 180=fast).
- estimated_key: Musical key (e.g., C, Gm, F#m).
- mfcc_mean: Coefficients representing the overall timbre/sound quality (complex patterns, no direct interpretation needed here).
- rms_mean: Average energy, related to perceived loudness/intensity (higher values mean louder overall).
- spectral_centroid_mean: Indicates the 'brightness' of the sound (higher values = brighter, more high-frequency content like cymbals or synths; lower values = darker, more bass content).
- zcr_mean: Zero-crossing rate, related to noisiness or percussiveness (higher values often indicate more noise or sharp attacks like drums).

Infer the following based SOLELY on the provided features:
- primary_genre: The single most likely main genre (Choose ONE from: Electronic, Rock, Pop, Hip-Hop, Jazz, Classical, Folk, Blues, RnB, Metal, Punk, World, Other).
- sub_genres: A list of 1-3 specific sub-genres fitting the primary genre (e.g., Synthwave, Indie Rock, Deep House, Bebop, Baroque, Boom Bap, Singer-Songwriter, Heavy Metal, Post-Punk).
- mood_tags: A list of 2-5 descriptive mood keywords (e.g., Energetic, Melancholic, Relaxing, Uplifting, Intense, Dreamy, Nostalgic, Aggressive, Peaceful, Dark, Happy, Sad, Romantic).
- instruments_likely: A list of 2-4 instruments likely present based on timbre, genre, and features (e.g., Synthesizer, Electric Guitar, Acoustic Drum Kit, Piano, Vocals, Bass Guitar, Strings, Brass Section, Woodwinds, Drum Machine, Acoustic Guitar).

Return ONLY a valid JSON object containing these four keys exactly as specified below. Do not include any introductory text, explanations, code formatting (like ```json), or concluding remarks.

Example JSON format:
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
        logger.error("Ollama client not available. Cannot tag track.")
        return None
    if not isinstance(features, dict) or not features:
         logger.error("Invalid or empty features received for tagging.")
         return None

    # Construct the user part of the prompt using the features
    # Ensure features are formatted clearly for the LLM
    user_prompt = f"Analyze the following audio features and generate the semantic tags according to the JSON format specified in the system instructions:\n\nFEATURES:\n{json.dumps(features, indent=2)}"

    try:
        logger.info(f"Sending tag request to Ollama model: {OLLAMA_MODEL}...")
        response = client.generate(
            model=OLLAMA_MODEL,
            system=SYSTEM_PROMPT, # Use the dedicated system parameter
            prompt=user_prompt,
            format="json", # Instruct Ollama to ensure the output is JSON
            options={"temperature": 0.3} # Lower temperature for more deterministic tags
        )
        logger.info("Received tag response from Ollama.")

        # The response content is in response['response']
        response_text = response.get("response", "")
        logger.debug(f"Raw tag response text:\n{response_text}") # Debug level for potentially large output

        # Attempt to parse the JSON string from the response
        tags = json.loads(response_text)

        # Basic validation: Check if all expected keys are present
        expected_keys = {"primary_genre", "sub_genres", "mood_tags", "instruments_likely"}
        if not expected_keys.issubset(tags.keys()):
             logger.warning(f"Generated tags JSON is missing expected keys. Got: {tags.keys()}")
             # Decide if you want to return partial data or None
             # return None # Stricter approach
             return tags # Lenient approach for now

        return tags

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Ollama: {e}", exc_info=True)
        logger.error(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        # Catch potential issues with the Ollama library call itself
        logger.error(f"Error during Ollama API call for tagging: {e}", exc_info=True)
        return None

# --- Main Execution Block (for direct script running) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate semantic tags for a music audio file using Ollama.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., tests/data/120bpm_D.wav)")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)

    if not audio_path.is_file():
        logger.error(f"Error: Audio file not found at {audio_path}")
    elif client is None:
         logger.error("Cannot proceed without a connection to Ollama.")
    else:
        logger.info(f"--- Starting Tagging Process for: {audio_path.name} ---")
        logger.info(f"1. Extracting features from: {audio_path}...")
        # Ensure analyzer is using the updated feature extraction logic
        features = extract_features(audio_path)

        if features:
            # Log extracted features for review (consider logging only keys or truncating values if too long)
            logger.info("2. Features extracted successfully:")
            logger.info(json.dumps(features, indent=2))

            logger.info("3. Attempting to generate tags via Ollama...")
            generated_tags = tag_track(features)

            if generated_tags:
                logger.info("4. Tagging successful!")
                logger.info("\n--- Generated Tags ---")
                logger.info(json.dumps(generated_tags, indent=2))
                logger.info("--- End Tagging Process ---")
            else:
                logger.error("4. Failed to generate tags for the audio file.")
                logger.info("--- End Tagging Process (with errors) ---")
        else:
            logger.error("Failed to extract features. Cannot proceed with tagging.")
            logger.info("--- End Tagging Process (feature extraction failed) ---")