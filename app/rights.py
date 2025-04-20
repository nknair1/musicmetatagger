# app/rights.py
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose a lightweight sentence transformer model
# More models: https://www.sbert.net/docs/pretrained_models.html
MODEL_NAME = 'all-MiniLM-L6-v2' # Good balance of speed and quality
SIMILARITY_THRESHOLD = 0.80 # Cosine similarity threshold (adjust as needed)

# --- Load Model ---
try:
    # This will download the model on first run (~80MB)
    logger.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Sentence transformer model loaded.")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {MODEL_NAME}", exc_info=True)
    model = None

# --- Corpus Loading (Placeholder) ---
def load_corpus(corpus_dir: Path) -> list[str]:
    """
    Loads lyrics/text from .txt files in a directory.
    """
    logger.info(f"Loading corpus texts from: {corpus_dir}")
    corpus_texts = []
    if not corpus_dir.is_dir():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return []

    for txt_file in corpus_dir.glob("*.txt"): # Find all .txt files
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Read the whole file as one text for simplicity,
                # or split into paragraphs/chunks if needed for more granular comparison
                content = f.read()
                if content: # Avoid adding empty files
                    corpus_texts.append(content)
            logger.debug(f"Loaded text from: {txt_file.name}")
        except Exception as e:
            logger.error(f"Failed to read corpus file {txt_file.name}: {e}")

    if not corpus_texts:
        logger.warning(f"No .txt files found or loaded from {corpus_dir}")
    else:
        logger.info(f"Loaded {len(corpus_texts)} texts from corpus.")
    return corpus_texts

# --- Rights Check Function ---
def check_lyric_similarity(input_lyrics: str, corpus_texts: list[str]) -> list[dict]:
    """
    Compares input lyrics against a corpus using sentence embeddings.
    Returns a list of potential matches above the threshold.
    """
    if model is None:
        logger.error("Sentence transformer model not loaded. Cannot perform similarity check.")
        return []
    if not input_lyrics:
        logger.warning("Empty input lyrics provided for similarity check.")
        return []
    if not corpus_texts:
        logger.warning("Empty corpus provided for similarity check.")
        return []

    try:
        logger.info("Embedding input lyrics...")
        input_embedding = model.encode(input_lyrics, convert_to_tensor=True)

        logger.info(f"Embedding corpus ({len(corpus_texts)} texts)...")
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

        logger.info("Calculating cosine similarities...")
        # Compute cosine similarity between input and all corpus entries
        cosine_scores = util.cos_sim(input_embedding, corpus_embeddings)[0] # Get the first row (input vs all corpus)

        # --- Original code causing error ---
        # high_similarity_indices = np.where(cosine_scores >= SIMILARITY_THRESHOLD)[0]

        # --- Corrected code ---
        # Move scores tensor from MPS/GPU to CPU before using NumPy
        cosine_scores_cpu = cosine_scores.cpu()
        high_similarity_indices = np.where(cosine_scores_cpu >= SIMILARITY_THRESHOLD)[0]

        potential_matches = []
        # Find indices where similarity exceeds threshold
        # Note: We iterate using the indices found by np.where

        for idx in high_similarity_indices:
            match = {
                "corpus_index": int(idx),
                "corpus_text_preview": corpus_texts[idx][:100] + "...",  # Show preview
                # Get the score corresponding to the index from the CPU tensor
                "similarity_score": float(cosine_scores_cpu[idx])
            }
            potential_matches.append(match)
            logger.warning(
                f"Potential high similarity found: Score={match['similarity_score']:.4f} with Corpus Index={match['corpus_index']}")

        return potential_matches

    except Exception as e:
        logger.error("Error during similarity calculation", exc_info=True)
        return []

# Example Usage
if __name__ == '__main__':
    # --- IMPORTANT: How to get lyrics? ---
    # Option 1: Manually provide lyrics for a test file
    test_lyrics = """
    There's a lady who's sure all that glitters is gold
    And she's buying a stairway to heaven
    When she gets there she knows, if the stores are all closed
    With a word she can get what she came for
    """
    # Option 2: Integrate a lyrics fetching library (see below)
    # Option 3: Accept lyrics as input alongside the audio file

    # --- Load Corpus ---
    # Create a dummy 'corpus' folder in your project root for now
    corpus_directory = Path(__file__).parent.parent / "corpus"
    if not corpus_directory.exists():
        corpus_directory.mkdir()
        logger.info(f"Created dummy corpus directory: {corpus_directory}")
        # You should manually add some .txt files here from public domain sources
        # e.g., download from https://www.gutenberg.org/

    corpus = load_corpus(corpus_directory) # For now, uses the dummy list

    if corpus:
        logger.info(f"\nChecking similarity for test lyrics...")
        matches = check_lyric_similarity(test_lyrics, corpus)

        if matches:
            logger.info("\nPotential Matches Found:")
            logger.info(json.dumps(matches, indent=2))
        else:
            logger.info("\nNo significant similarity found above threshold.")
    else:
        logger.warning("Could not load corpus for testing.")