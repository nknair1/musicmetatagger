# app/rights.py
import os
import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from ollama import Client # Import Ollama client

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Sentence Transformer Configuration ---
# Choose a lightweight sentence transformer model
MODEL_NAME = 'all-MiniLM-L6-v2' # Good balance of speed and quality
SIMILARITY_THRESHOLD = 0.80 # Cosine similarity threshold (adjust as needed)

# --- Load Sentence Transformer Model ---
try:
    # This will download the model on first run (~80MB)
    logger.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Sentence transformer model loaded.")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {MODEL_NAME}", exc_info=True)
    model = None

# --- LLM Client for Risk Assessment ---
# Initialize Ollama client for the risk assessment function
try:
    # Use environment variables or defaults
    OLLAMA_HOST_RIGHTS = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_client = Client(host=OLLAMA_HOST_RIGHTS)
    llm_client.list() # Verify connection quickly
    logger.info(f"Ollama client initialized successfully for rights assessment at {OLLAMA_HOST_RIGHTS}.")
except Exception as e:
    logger.error(f"Failed to initialize Ollama client for rights assessment: {e}", exc_info=True)
    llm_client = None

# --- LLM Prompt for Risk Assessment ---
RIGHTS_ASSESSMENT_PROMPT = """You are a cautious copyright assistant reviewing potential text similarity matches. You are given the original input lyrics and one or more snippets from a corpus that showed high semantic similarity (> {threshold:.2f}) using sentence embeddings. Your task is to provide a brief risk assessment based ONLY on the provided text snippets.

Analyze the following:
1.  **Verbatim Overlap**: Does the input lyrics snippet appear to contain near-exact phrases or sentences found in the corpus snippet?
2.  **Substantiality**: Is the overlapping part significant (e.g., multiple lines, distinctive phrases) or just a few common words?
3.  **Context**: Based purely on the text, does the similarity seem coincidental (common phrasing) or potentially indicative of copying? (Be conservative).

Based on this analysis, provide a concise risk assessment. Return ONLY a valid JSON object with the following keys:
- "overall_risk_level": (string) Choose ONE: "Low", "Medium", "High", "Undetermined".
- "assessment_notes": (string) A brief explanation (1-2 sentences) justifying the risk level, mentioning verbatim overlap or substantiality if observed.

Example JSON Output:
{{  # Escaped brace
  "overall_risk_level": "Medium",
  "assessment_notes": "Some verbatim phrase overlap detected between input and corpus snippet index {{index}}. Similarity seems more than coincidental common phrasing."
  # Note: {{index}} is also escaped, though it won't be filled by .format anyway
}}  # Escaped brace
OR
{{  # Escaped brace
  "overall_risk_level": "Low",
  "assessment_notes": "High semantic similarity score noted, but no significant verbatim overlap found in the provided snippets. May be thematic similarity."
}}  # Escaped brace

Input Lyrics Snippet:
```
{input_lyrics_snippet}
```

Potential Match(es) from Corpus:
```
{corpus_matches_formatted}
```

Provide your JSON assessment:"""


# --- Corpus Loading Function ---
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
                content = f.read()
                if content and content.strip(): # Avoid adding empty/whitespace files
                    corpus_texts.append(content)
            # logger.debug(f"Loaded text from: {txt_file.name}") # Can be noisy
        except Exception as e:
            logger.error(f"Failed to read corpus file {txt_file.name}: {e}")

    if not corpus_texts:
        logger.warning(f"No non-empty .txt files found or loaded from {corpus_dir}")
    else:
        logger.info(f"Loaded {len(corpus_texts)} texts from corpus.")
    return corpus_texts


# --- Sentence Transformer Similarity Check Function ---
def check_lyric_similarity(input_lyrics: str, corpus_texts: list[str]) -> list[dict]:
    """
    Compares input lyrics against a corpus using sentence embeddings.
    Returns a list of potential matches above the threshold.
    """
    if model is None:
        logger.error("Sentence transformer model not loaded. Cannot perform similarity check.")
        return []
    if not input_lyrics or not input_lyrics.strip():
        logger.warning("Empty or whitespace-only input lyrics provided for similarity check.")
        return []
    if not corpus_texts:
        logger.warning("Empty corpus provided for similarity check.")
        return []

    try:
        logger.info("Embedding input lyrics...")
        # Ensure input is treated as a single document/string for embedding
        input_embedding = model.encode(input_lyrics, convert_to_tensor=True)

        logger.info(f"Embedding corpus ({len(corpus_texts)} texts)...")
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

        logger.info("Calculating cosine similarities...")
        # Compute cosine similarity between input (1 embedding) and all corpus entries (N embeddings)
        # Result shape: (1, N)
        cosine_scores = util.cos_sim(input_embedding, corpus_embeddings)

        # Ensure we handle the tensor correctly (it might be nested if input was list)
        if cosine_scores.dim() > 1:
            cosine_scores = cosine_scores[0] # Get the first row if necessary

        # Move scores tensor from MPS/GPU to CPU before using NumPy
        cosine_scores_cpu = cosine_scores.cpu().numpy() # Convert to numpy array on CPU

        potential_matches = []
        # Find indices where similarity exceeds threshold using the numpy array
        high_similarity_indices = np.where(cosine_scores_cpu >= SIMILARITY_THRESHOLD)[0]

        for idx in high_similarity_indices:
            if 0 <= idx < len(corpus_texts): # Bounds check
                match = {
                    "corpus_index": int(idx),
                    "corpus_text_preview": corpus_texts[idx][:100] + "...", # Show preview
                    # Get the score corresponding to the index from the CPU numpy array
                    "similarity_score": float(cosine_scores_cpu[idx])
                }
                potential_matches.append(match)
                # Log warning for high similarity found
                logger.warning(f"Potential high similarity found: Score={match['similarity_score']:.4f} with Corpus Index={match['corpus_index']}")
            else:
                logger.error(f"Index {idx} out of bounds for corpus_texts (size {len(corpus_texts)})")


        if not potential_matches:
             logger.info("No similarity scores found above threshold.")

        return potential_matches

    except Exception as e:
        logger.error("Error during similarity calculation", exc_info=True)
        return []


# --- LLM Risk Assessment Function ---
def assess_rights_risk_llm(input_lyrics: str, matches: list[dict], corpus_texts: list[str]) -> dict | None:
    """
    Uses an LLM to provide a qualitative risk assessment based on similarity matches.
    'matches' is the list returned by check_lyric_similarity.
    'corpus_texts' is the full list of texts loaded from the corpus.
    """
    if llm_client is None:
        logger.error("Ollama client not available for risk assessment.")
        return {"error": "LLM client unavailable."}
    if not matches:
        # This function shouldn't normally be called without matches, but handle defensively
        logger.info("No similarity matches provided, skipping LLM risk assessment.")
        return None # No assessment needed if no matches

    # Prepare context for the LLM
    input_snippet = input_lyrics[:500] + ("..." if len(input_lyrics) > 500 else "") # Limit input snippet

    formatted_matches = []
    match_indices_used = set() # Track indices to avoid duplicate snippets if multiple matches point to same text

    for match in matches:
        corpus_index = match.get("corpus_index", -1)
        if 0 <= corpus_index < len(corpus_texts) and corpus_index not in match_indices_used:
             corpus_snippet = corpus_texts[corpus_index][:500] + ("..." if len(corpus_texts[corpus_index]) > 500 else "") # Limit corpus snippet
             formatted_matches.append(
                 f"Corpus Index: {corpus_index}\n"
                 f"Similarity Score: {match.get('similarity_score', 'N/A'):.4f}\n"
                 f"Corpus Snippet:\n---\n{corpus_snippet}\n---"
             )
             match_indices_used.add(corpus_index)
        elif corpus_index in match_indices_used:
             logger.debug(f"Skipping duplicate snippet for corpus index {corpus_index} in LLM prompt.")
        else:
             # Log invalid index but maybe don't include in prompt
             logger.warning(f"Invalid corpus index {corpus_index} found in matches list.")

    if not formatted_matches:
         logger.warning("Could not format any valid matches for LLM assessment.")
         return {"error": "Failed to prepare match context for LLM."}

    corpus_matches_str = "\n\n".join(formatted_matches)

    # Format the final prompt, ensuring the threshold value is included
    prompt = RIGHTS_ASSESSMENT_PROMPT.format(
        threshold=SIMILARITY_THRESHOLD,
        input_lyrics_snippet=input_snippet,
        corpus_matches_formatted=corpus_matches_str
    )

    try:
        logger.info("Sending risk assessment request to Ollama...")
        response = llm_client.generate(
            model=os.getenv("OLLAMA_MODEL", "llama3:8b"), # Ensure consistent model usage
            prompt=prompt,
            format="json",
            options={"temperature": 0.2} # Low temperature for less creative risk analysis
        )
        logger.info("Received risk assessment response from Ollama.")
        response_text = response.get("response", "")
        logger.debug(f"Raw LLM risk assessment response: {response_text}")

        assessment_json = json.loads(response_text)
        # Basic validation
        if "overall_risk_level" in assessment_json and "assessment_notes" in assessment_json:
            logger.info(f"LLM Risk Assessment: {assessment_json.get('overall_risk_level')}")
            return assessment_json
        else:
            logger.warning(f"LLM risk assessment response missing required keys. Raw: {response_text}")
            return {"error": "LLM response format invalid.", "raw_response": response_text}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON risk assessment response from Ollama: {e}", exc_info=True)
        logger.error(f"Raw response was: {response_text}")
        return {"error": "LLM response JSON decode failed.", "raw_response": response_text}
    except Exception as e:
        logger.error(f"Error during Ollama API call for risk assessment: {e}", exc_info=True)
        return {"error": "Ollama API call failed during risk assessment."}

# --- Example Usage (for testing rights.py directly) ---
if __name__ == '__main__':
    # Configure logging for standalone run if needed
    # logging.basicConfig(level=logging.DEBUG) # Set to DEBUG to see more logs

    # --- IMPORTANT: How to get lyrics? ---
    test_lyrics = """
    There's a lady who's sure all that glitters is gold
    And she's buying a stairway to heaven
    When she gets there she knows, if the stores are all closed
    With a word she can get what she came for
    Ooh, ooh, and she's buying a stairway to heaven
    """
    # Example lyrics with potential overlap with a corpus item
    test_lyrics_overlap = """
    Call me Ishmael. Some years ago - never mind how long precisely - having
    little or no money in my purse, and nothing particular to interest me
    on shore, I thought I would sail about a little and see the watery part
    of the world.
    """

    # --- Load Corpus ---
    # Assumes 'corpus' folder is in the parent directory of 'app'
    corpus_directory = Path(__file__).parent.parent / "corpus"
    if not corpus_directory.exists():
         logger.error(f"Corpus directory not found for standalone test: {corpus_directory}")
         corpus = []
    else:
        corpus = load_corpus(corpus_directory)

    if corpus and model is not None: # Check if model loaded
        logger.info(f"\n--- Checking similarity for standard test lyrics ---")
        matches_standard = check_lyric_similarity(test_lyrics, corpus)
        if matches_standard:
            logger.info("Matches found, attempting LLM risk assessment...")
            risk_assessment = assess_rights_risk_llm(test_lyrics, matches_standard, corpus)
            logger.info(f"\nLLM Risk Assessment Result (Standard Lyrics):")
            logger.info(json.dumps(risk_assessment, indent=2))
        else:
             logger.info("No high similarity matches found for standard lyrics.")


        logger.info(f"\n--- Checking similarity for overlap test lyrics ---")
        matches_overlap = check_lyric_similarity(test_lyrics_overlap, corpus)
        if matches_overlap:
            logger.info("Matches found, attempting LLM risk assessment...")
            risk_assessment_overlap = assess_rights_risk_llm(test_lyrics_overlap, matches_overlap, corpus)
            logger.info(f"\nLLM Risk Assessment Result (Overlap Lyrics):")
            logger.info(json.dumps(risk_assessment_overlap, indent=2))
        else:
             logger.info("No high similarity matches found for overlap lyrics.")

    elif model is None:
         logger.error("Sentence transformer model failed to load. Cannot run tests.")
    else: # Corpus is empty
        logger.warning("Could not load corpus for standalone testing.")