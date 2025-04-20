# app/metadata.py
from ollama import Client
import os
import json
import logging # Add logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

try:
    client = Client(host=OLLAMA_HOST)
except Exception as e:
    logger.error(f"Error connecting to Ollama at {OLLAMA_HOST}: {e}", exc_info=True)
    client = None

# --- Prompt Engineering ---
# Note: You might need separate prompts if generating bios/album context later
METADATA_SYSTEM_PROMPT = """You are an expert music copywriter creating engaging descriptions for streaming platforms and SEO. Given semantic tags for a music track, generate the following:

1.  **track_description**: A concise (2-3 sentences, ~250 characters max), evocative description highlighting the genre, mood, and instrumentation. Make it sound appealing for discovery playlists.
2.  **seo_keywords**: A list of 5-10 relevant keywords (strings) for search engines, based on genre, sub-genres, mood, and instruments.

Return ONLY a valid JSON object containing these two keys: 'track_description' and 'seo_keywords'. Do not include any other text, explanations, or markdown formatting.
Example JSON format:
{
  "track_description": "Dive into a nostalgic soundscape with this Synthwave track. Dreamy synthesizers and a steady drum machine beat create a relaxing yet driving atmosphere, perfect for late-night coding sessions or cruising.",
  "seo_keywords": ["synthwave", "electronic", "80s", "retro", "chillwave", "nostalgic", "dreamy", "relaxing", "instrumental", "synthesizer"]
}"""

# --- Metadata Function ---
def build_track_metadata(tags: dict) -> dict | None:
    """
    Takes semantic tags, generates descriptive metadata using Ollama.
    Returns None if client isn't available or parsing fails.
    """
    if client is None:
        logger.error("Ollama client not available. Cannot build metadata.")
        return None
    if not isinstance(tags, dict) or not tags:
         logger.error("Invalid or empty tags received for metadata generation.")
         return None

    # Construct the user part of the prompt using the tags
    # Ensure tags are formatted nicely for the LLM
    user_prompt = f"Generate metadata for a track with the following characteristics:\n\nTAGS:\n{json.dumps(tags, indent=2)}"

    try:
        logger.info(f"Sending metadata request to Ollama model: {OLLAMA_MODEL}...")
        response = client.generate(
            model=OLLAMA_MODEL,
            system=METADATA_SYSTEM_PROMPT,
            prompt=user_prompt,
            format="json" # Request JSON format
        )
        logger.info("Received metadata response from Ollama.")
        response_text = response.get("response", "")
        logger.debug(f"Raw metadata response text:\n{response_text}")

        metadata = json.loads(response_text)
        # Basic validation (can be improved with schemas like Pydantic)
        if "track_description" in metadata and "seo_keywords" in metadata:
             return metadata
        else:
             logger.error(f"Metadata JSON missing required keys. Response: {response_text}")
             return None


    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON metadata response from Ollama: {e}", exc_info=True)
        logger.error(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        logger.error(f"Error during Ollama API call for metadata", exc_info=True)
        return None

# Example usage (for testing this file directly)
if __name__ == '__main__':
    # Example tags (replace with actual output from tagger.py)
    sample_tags = {
        "primary_genre": "Electronic",
        "sub_genres": ["Synthwave", "Chillwave"],
        "mood_tags": ["Nostalgic", "Dreamy", "Relaxing"],
        "instruments_likely": ["Synthesizer", "Drum Machine", "Bass Guitar"]
    }

    logger.info("Attempting to generate metadata...")
    generated_metadata = build_track_metadata(sample_tags)

    if generated_metadata:
        logger.info("\nGenerated Metadata:")
        logger.info(json.dumps(generated_metadata, indent=2))
    else:
        logger.error("Failed to generate metadata.")