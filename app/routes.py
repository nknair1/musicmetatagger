# app/routes.py
import os
import json
import logging
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, current_app

# Assuming these modules are in the same 'app' directory
# or the project structure allows importing them
try:
    from . import analyzer
    from . import tagger
    from . import metadata
    from . import rights
except ImportError as e:
     # Fallback for running flask directly, though using 'python -m flask run' is better
    import analyzer
    import tagger
    import metadata
    import rights
    logging.warning(f"ImportWarning: Could not use relative imports: {e}. Using direct imports.")


# Configure logger for the Flask app if not already configured
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FlaskAPI")

# --- Create Flask App ---
# Using Flask(__name__) is standard practice
# We can create it here or use an app factory pattern later if it gets complex
app = Flask(__name__)

# --- Helper Function (Optional but Recommended) ---
# This encapsulates the logic from main.py's process_track
def run_full_pipeline(audio_file_path: Path, lyrics_text: str | None) -> dict | None:
    """
    Runs the analysis pipeline, similar to main.py's process_track.
    Returns the results dictionary or None on failure.
    """
    logger.info(f"--- API: Starting Full Analysis for: {audio_file_path.name} ---")
    pipeline_results = {}

    # Ensure Ollama is reachable (add check if needed)
    if not hasattr(tagger, 'client') or tagger.client is None:
        logger.error("API Error: Ollama client not available.")
        return {"error": "Ollama connection failed. Ensure server is running."}

    # 1. Extract Features
    logger.info("API Step 1: Extracting audio features...")
    features = analyzer.extract_features(audio_file_path)
    if not features:
        logger.error("API Error: Failed to extract features.")
        return {"error": "Feature extraction failed."}
    pipeline_results["audio_features"] = features
    logger.info("API: Features extracted.")

    # 2. Generate Semantic Tags
    logger.info("API Step 2: Generating semantic tags...")
    tags = tagger.tag_track(features)
    if not tags:
        logger.warning("API Warning: Failed to generate tags. Metadata/Risk assessment might be affected.")
        pipeline_results["semantic_tags"] = None
    else:
        pipeline_results["semantic_tags"] = tags
        logger.info("API: Tags generated.")

    # 3. Generate Metadata Description
    logger.info("API Step 3: Generating descriptive metadata...")
    if tags:
        desc_metadata = metadata.build_track_metadata(tags)
        pipeline_results["descriptive_metadata"] = desc_metadata if desc_metadata else {"warning": "Metadata generation failed"}
        logger.info("API: Metadata generated (or attempt failed).")
    else:
        pipeline_results["descriptive_metadata"] = {"info": "Skipped, tags not available"}
        logger.info("API: Metadata skipped (no tags).")


    # 4. Check Lyric Similarity & Risk (if lyrics provided)
    if lyrics_text and lyrics_text.strip():
        logger.info("API Step 4: Checking lyric similarity and assessing risk...")
        # Define default corpus path relative to project root
        corpus_path = Path(__file__).parent.parent / "corpus" # Assumes 'app' is one level down
        if not corpus_path.is_dir():
            logger.error(f"API Error: Corpus directory not found at {corpus_path}")
            pipeline_results["lyric_analysis"] = {"error": f"Corpus directory missing: {corpus_path}"}
        else:
             corpus_texts = rights.load_corpus(corpus_path)
             if not corpus_texts:
                  pipeline_results["lyric_analysis"] = {"status": "No corpus loaded"}
             else:
                  # Perform similarity check
                  matches = rights.check_lyric_similarity(lyrics_text, corpus_texts)

                  # --- Call the LLM Risk Assessment (Enhancement 2) ---
                  risk_assessment = None
                  if matches: # Only assess risk if potential similarity is found
                       logger.info("API Step 4b: Assessing copyright risk via LLM...")
                       risk_assessment = rights.assess_rights_risk_llm(lyrics_text, matches, corpus_texts)
                  # --- End Enhancement 2 ---

                  pipeline_results["lyric_analysis"] = {
                      "input_lyrics_preview": lyrics_text[:100] + "...",
                      "similarity_threshold": rights.SIMILARITY_THRESHOLD,
                      "potential_matches": matches,
                      "llm_risk_assessment": risk_assessment # Add the assessment here
                  }
        logger.info("API: Lyric analysis completed.")

    else:
        pipeline_results["lyric_analysis"] = None # No lyrics provided
        logger.info("API Step 4: Skipping lyric analysis (no lyrics provided).")

    logger.info(f"--- API: Full Analysis Complete ---")
    return pipeline_results


# --- API Endpoint ---
@app.route("/process", methods=["POST"])
def process_audio_endpoint():
    """
    API endpoint to process an uploaded audio file and optional lyrics.
    Expects 'audio_file' in request.files and optionally 'lyrics' in request.form.
    """
    logger.info("Received request to /process endpoint.")

    # Check if audio file is present
    if 'audio_file' not in request.files:
        logger.error("Request missing 'audio_file' part.")
        return jsonify({"error": "Missing 'audio_file' in request files"}), 400

    audio_file = request.files['audio_file']

    # Check if filename is present
    if audio_file.filename == '':
        logger.error("No selected audio file.")
        return jsonify({"error": "No selected audio file"}), 400

    # Get lyrics from form data (optional)
    lyrics_content = request.form.get('lyrics', None) # Use .get for optional field
    if lyrics_content:
        logger.info("Lyrics provided in request form.")
    else:
        logger.info("No lyrics provided in request form.")


    # Save the uploaded file temporarily because librosa needs a path
    # Using tempfile is safer as it handles cleanup
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_audio_file:
            audio_file.save(tmp_audio_file.name)
            tmp_audio_path = Path(tmp_audio_file.name)
            logger.info(f"Audio file saved temporarily to: {tmp_audio_path}")

        # Run the full pipeline using the helper function
        results = run_full_pipeline(tmp_audio_path, lyrics_content)

    except Exception as e:
         logger.error(f"Error during pipeline execution: {e}", exc_info=True)
         return jsonify({"error": "An internal error occurred during processing."}), 500
    finally:
        # Clean up the temporary file
        if 'tmp_audio_path' in locals() and tmp_audio_path.exists():
            try:
                os.unlink(tmp_audio_path)
                logger.info(f"Temporary audio file deleted: {tmp_audio_path}")
            except Exception as e_unlink:
                logger.error(f"Error deleting temporary file {tmp_audio_path}: {e_unlink}")

    # Return results or error message
    if results and "error" not in results:
        return jsonify(results), 200
    elif results and "error" in results:
         # Return specific errors if available and safe to expose
         return jsonify(results), 500 # Internal server error likely
    else:
         # Catch all for unexpected pipeline failure
         return jsonify({"error": "Pipeline failed to produce results."}), 500


# --- Simple Root Endpoint (Optional) ---
@app.route("/")
def index():
    # You could return basic HTML instructions here later
    return "MusicMetaTagger API is running. Use POST /process endpoint."


# --- Allow running directly for development (though 'flask run' is preferred) ---
if __name__ == "__main__":
    # Note: Running this way might have issues with relative imports if not careful
    # Use 'export FLASK_APP=app.routes; python -m flask run --debug' instead
    logger.warning("Running Flask app directly. Use 'flask run' for better development experience.")
    app.run(debug=True, port=5000) # Port 5000 is common for Flask dev