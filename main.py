# main.py
import argparse
import json
import logging
from pathlib import Path
import sys

# --- Setup Logging ---
# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MusicMetaTaggerCLI") # Create a logger for this script

# --- Import Application Modules ---
# Add the 'app' directory to the Python path to allow imports
# This makes the script runnable from the project root directory
project_root = Path(__file__).parent
app_dir = project_root / "app"
sys.path.insert(0, str(project_root)) # Add project root first

try:
    from app import analyzer, tagger, metadata, rights
except ImportError as e:
    logger.error(f"Error importing application modules: {e}")
    logger.error("Ensure this script is run from the project root directory (`MusicMetaTagger/`)")
    logger.error("And that the 'app' directory with __init__.py exists.")
    sys.exit(1) # Exit if imports fail

# --- Main Function ---
def process_track(audio_path: Path, lyrics: str | None = None, corpus_dir: Path | None = None):
    """
    Runs the full analysis pipeline for a single audio track.
    """
    logger.info(f"--- Starting Full Analysis for: {audio_path.name} ---")
    full_results = {}

    # 1. Extract Features
    logger.info("Step 1: Extracting audio features...")
    features = analyzer.extract_features(audio_path)
    if not features:
        logger.error("Failed to extract features. Aborting analysis.")
        return None
    full_results["audio_features"] = features
    logger.info("Features extracted successfully.")
    # logger.debug(json.dumps(features, indent=2)) # Log features only if debugging

    # 2. Generate Semantic Tags
    logger.info("Step 2: Generating semantic tags via LLM...")
    tags = tagger.tag_track(features)
    if not tags:
        logger.warning("Failed to generate semantic tags. Proceeding without them.")
        full_results["semantic_tags"] = None
    else:
        full_results["semantic_tags"] = tags
        logger.info("Semantic tags generated successfully.")
        # logger.info(json.dumps(tags, indent=2))

    # 3. Generate Metadata Description
    logger.info("Step 3: Generating descriptive metadata via LLM...")
    if tags: # Only generate metadata if tags exist
        desc_metadata = metadata.build_track_metadata(tags)
        if not desc_metadata:
            logger.warning("Failed to generate descriptive metadata.")
            full_results["descriptive_metadata"] = None
        else:
            full_results["descriptive_metadata"] = desc_metadata
            logger.info("Descriptive metadata generated successfully.")
            # logger.info(json.dumps(desc_metadata, indent=2))
    else:
        logger.warning("Skipping metadata generation because tags are missing.")
        full_results["descriptive_metadata"] = None


    # 4. Check Lyric Similarity (if lyrics provided)
    logger.debug(f"Inside process_track - Received lyrics type: {type(lyrics)}, content preview: '{str(lyrics)[:50]}...'")
    # --- End debug log ---

    if lyrics and lyrics.strip(): # Check if lyrics is not None AND not just whitespace
        logger.info("Step 4: Checking lyric similarity...")
        # ... (rest of the lyric check logic remains the same) ...
        # Load corpus dynamically here if needed, or assume rights module handles it
        corpus_texts = rights.load_corpus(corpus_dir) # Assumes load_corpus is efficient enough
        if not corpus_texts:
            logger.warning("Corpus is empty. Cannot perform similarity check.")
            full_results["lyric_analysis"] = {"status": "No corpus loaded"}
        else:
            matches = rights.check_lyric_similarity(lyrics, corpus_texts)
            full_results["lyric_analysis"] = {
                "input_lyrics_preview": lyrics[:100] + "...",
                "similarity_threshold": rights.SIMILARITY_THRESHOLD,
                "potential_matches": matches
            }
            logger.info(f"Lyric similarity check completed. Found {len(matches)} potential match(es).")
    else:
        # Log why it's being skipped
        if lyrics is None:
            logger.info("Step 4: Skipping lyric similarity check (no lyrics provided).")
        else:
            logger.info("Step 4: Skipping lyric similarity check (provided lyrics file was empty or whitespace).")
        full_results["lyric_analysis"] = None # Indicate no analysis was performed


    logger.info(f"--- Full Analysis Complete for: {audio_path.name} ---")
    return full_results

# --- Command Line Interface Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Music Metadata Generator CLI")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file (e.g., tests/data/120bpm_D.wav)")
    parser.add_argument("-l", "--lyrics_file", type=str, help="Optional: Path to a text file containing the lyrics.", default=None)
    parser.add_argument("-c", "--corpus_dir", type=str, help="Path to the directory containing corpus text files for rights check.", default="corpus") # Default to 'corpus' folder
    parser.add_argument("-o", "--output_file", type=str, help="Optional: Path to save the full results as a JSON file.", default=None)

    args = parser.parse_args()

    # --- Prepare Inputs ---
    audio_input_path = Path(args.audio_file)
    if not audio_input_path.is_file():
        logger.error(f"Audio file not found: {audio_input_path}")
        sys.exit(1)

    input_lyrics_text = None
    if args.lyrics_file:
        lyrics_input_path = Path(args.lyrics_file)
        if not lyrics_input_path.is_file():
            logger.warning(f"Lyrics file not found: {lyrics_input_path}. Proceeding without lyrics.")
        else:
            try:
                with open(lyrics_input_path, 'r', encoding='utf-8') as f:
                    input_lyrics_text = f.read()
                logger.info(f"Loaded lyrics from: {lyrics_input_path}")
            except Exception as e:
                logger.error(f"Failed to read lyrics file {lyrics_input_path}: {e}")
                # Decide if you want to exit or continue without lyrics
                logger.warning("Proceeding without lyrics due to file read error.")


    corpus_input_dir = Path(args.corpus_dir)

    # --- Run Processing ---
    # Ensure Ollama server is running before starting
    logger.info("Please ensure the Ollama server is running in the background (`ollama serve`).")
    # Add a check for Ollama connection if possible (e.g., calling tagger.client.list())
    if hasattr(tagger, 'client') and tagger.client is not None:
        try:
            tagger.client.list() # Quick check
            logger.info("Ollama connection verified.")
        except Exception as e:
            logger.error(f"Failed to verify Ollama connection: {e}")
            logger.error("Cannot proceed without Ollama. Please start the Ollama server.")
            sys.exit(1)
    elif not hasattr(tagger, 'client') or tagger.client is None:
         logger.error("Ollama client not initialized in tagger module. Cannot proceed.")
         sys.exit(1)


    results = process_track(audio_input_path, input_lyrics_text, corpus_input_dir)

    # --- Output Results ---
    if results:
        logger.info("\n--- Combined Results ---")
        # Pretty print the JSON results to the console
        print(json.dumps(results, indent=2))

        # Save to file if requested
        if args.output_file:
            output_path = Path(args.output_file)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Full results saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_path}: {e}")
    else:
        logger.error("Analysis failed to produce results.")