# AI-Powered Music Metadata Generation and Semantic Tagging Engine

## Overview & Motivation

This project explores the capabilities and potential limits of using Large Language Models (LLMs) and Natural Language Processing (NLP) techniques for analyzing music audio and generating relevant metadata. The primary goal is to automatically extract audio features, use an LLM (specifically Llama 3 running locally via Ollama) to generate semantic tags and descriptive text, and perform a basic lyric similarity check with an LLM-based risk assessment.

**Motivation:** This project was undertaken primarily as a personal exploration ("for fun") to test the boundaries of what could be achieved rapidly with modern AI tools, particularly focusing on prompt engineering for music-related tasks within a zero-budget constraint.

**AI Collaboration:** The development process involved significant interaction and iterative refinement with AI assistants, including models like OpenAI's GPT series (Specifically O3), Google's Gemini series(Specifically Gemini 2.5 Pro), and Anthropic's Claude series(Specifically Claude 3.7 Sonnet Thinking accessed via GitHub Copilot). Their assistance was instrumental in debugging, brainstorming, code generation, and structuring the project. This project serves as an example of human-AI collaboration in software development.

**Disclaimer:** This is an experimental project. The accuracy of the generated tags, metadata, and especially the rights assessment depends heavily on the performance of the local LLM, the quality of the prompts, and the limited scope of the rights checking corpus. It is not intended for production use without significant further development and validation.

## Core Features

* **Audio Feature Extraction:** Uses `librosa` to extract features like tempo (BPM), estimated key root, MFCCs (timbre), RMS energy (loudness), spectral centroid (brightness), and zero-crossing rate.
* **LLM-Powered Semantic Tagging:** Leverages a local LLM (Llama 3 via Ollama) with engineered prompts to generate semantic tags based on extracted audio features, including:
    * Primary Genre
    * Sub-genres
    * Mood Tags
    * Likely Instruments
* **LLM-Powered Metadata Generation:** Uses the LLM to generate structured metadata based on the generated tags:
    * SEO-friendly track descriptions
    * Relevant SEO keywords
* **Basic Lyric Rights Check:**
    * Uses `sentence-transformers` to calculate cosine similarity between input lyrics and texts in a local corpus.
    * Flags potential matches above a defined threshold (default: 0.80).
    * **LLM Risk Assessment:** If high similarity is detected, uses the LLM again to provide a qualitative risk assessment (Low/Medium/High) based on the matched text snippets.

## Tech Stack

* **Language:** Python 3.10+
* **Core Libraries:**
    * `librosa`: Audio analysis and feature extraction
    * `ollama-python`: Interacting with local Ollama server
    * `sentence-transformers`: Lyric embedding and similarity calculation
    * `numpy`: Numerical operations
    * `Flask`: Web framework for the API
* **LLM:** Meta Llama 3 (8B model recommended) run locally via [Ollama](https://ollama.com/)
* **Testing:** `pytest`, `pytest-mock`
* **Environment:** Virtual environment (`venv`)

## Folder Structure

```
MusicMetaTagger/
│
├── app/                 # Core application logic
│   ├── init.py
│   ├── analyzer.py      # Audio feature extraction
│   ├── tagger.py        # LLM semantic tagging
│   ├── metadata.py      # LLM metadata generation
│   ├── rights.py        # Lyric similarity & LLM risk assessment
│   └── routes.py        # Flask API endpoints
│
├── tests/               # Pytest unit tests
│   ├── data/            # Sample audio/lyrics for testing
│   │   ├── 120bpm_D.wav
│   │   └── lyrics.txt
│   ├── test_analyzer.py
│   ├── test_metadata.py
│   ├── test_rights.py
│   └── test_tagger.py
│
├── corpus/              # Directory for storing reference texts (e.g., public domain books)
│   ├── mobydick.txt     # Example
│   └── tale-of-two-cities.txt # Example
│
├── .venv/               # Virtual environment files (ignored by git)
├── main.py              # Command-line interface script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore           # Files/directories ignored by git
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nknair1/musicmetatagger.git
   cd musicmetatagger
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Setup Ollama and Llama 3:**
   * Download and install Ollama from [ollama.com](https://ollama.com/).
   * Pull the Llama 3 8B model (approx. 4.7GB):
     ```bash
     ollama pull llama3:8b
     ```
   * Start the Ollama server in a **separate terminal window** and keep it running:
     ```bash
     ollama serve
     ```
     *(Verify it's running by visiting `http://localhost:11434` in your browser or using `curl http://localhost:11434`)*

5. **Prepare Corpus Directory:**
   * The application looks for a `corpus/` directory in the project root by default for the rights check.
   * Place plain text files (`.txt`) containing reference texts (e.g., public domain books from Project Gutenberg) inside this directory. The quality and size of this corpus directly impact the rights check feature.

## Usage

Make sure your virtual environment is active (`source .venv/bin/activate`) and the Ollama server is running (`ollama serve`) before using the tool.

### 1. Command-Line Interface (`main.py`)

Run the analysis pipeline from your terminal.

**Basic Usage (Audio Only):**
```bash
python main.py path/to/your/audiofile.wav
```
*(Example: python main.py tests/data/120bpm_D.wav)*

**With Lyrics File:**
```bash
python main.py path/to/audio.mp3 -l path/to/lyrics.txt
```
*(Example: python main.py tests/data/120bpm_D.wav -l tests/data/lyrics.txt)*

**Specify Corpus Directory:**
```bash
python main.py audio.wav -l lyrics.txt -c /path/to/your/corpus_folder
```

**Save Output to JSON File:**
```bash
python main.py audio.wav -l lyrics.txt -o output_results.json
```

### 2. Flask API

Run a web server providing API endpoints.

**Start the Server:**
```bash
export FLASK_APP=app.routes
# Or on Windows: set FLASK_APP=app.routes
python -m flask run --debug
```
*(The server will typically run on http://127.0.0.1:5000/)*

**Send Request to /process Endpoint (using curl):**

*Audio Only:*
```bash
curl -X POST -F "audio_file=@path/to/your/audiofile.wav" http://127.0.0.1:5000/process
```
*(Example: curl -X POST -F "audio_file=@tests/data/120bpm_D.wav" http://127.0.0.1:5000/process)*

*Audio and Lyrics (from file):*
```bash
curl -X POST \
     -F "audio_file=@path/to/audio.wav" \
     -F "lyrics=<path/to/lyrics.txt" \
     http://127.0.0.1:5000/process
```
*(Example: curl -X POST -F "audio_file=@tests/data/120bpm_D.wav" -F "lyrics=<tests/data/lyrics.txt" http://127.0.0.1:5000/process)*

*Audio and Lyrics (as text):*
```bash
curl -X POST \
     -F "audio_file=@path/to/audio.wav" \
     -F "lyrics=These are the lyrics typed directly" \
     http://127.0.0.1:5000/process
```

## Example Output (JSON)

The tool (both CLI and API) outputs a JSON structure containing the analysis results:

```json
{
  "audio_features": {
    "tempo_bpm": 117.45,
    "estimated_key_root": "D",
    "mfcc_mean": [ -663.99, 1.77, /* ... more values */ -3.87 ],
    "rms_mean": 0.01126,
    "spectral_centroid_mean": 1056.56,
    "zcr_mean": 0.0496
  },
  "semantic_tags": {
    "primary_genre": "Electronic",
    "sub_genres": [ "Synthwave", "Chillout" ],
    "mood_tags": [ "Energetic", "Upbeat", "Catchy" ],
    "instruments_likely": [ "Synthesizer", "Drum Machine", "Bass Guitar" ]
  },
  "descriptive_metadata": {
    "track_description": "Get energized with this synth-driven Electronic track! Catchy synthesizers...",
    "seo_keywords": [ "electronic", "synthwave", /* ... more keywords */ ]
  },
  "lyric_analysis": {
    "input_lyrics_preview": "There's a lady who's sure all that glitters is gold...",
    "similarity_threshold": 0.8,
    "potential_matches": [
      {
        "corpus_index": 1, // Index in the loaded corpus list
        "corpus_text_preview": "There's a lady who's sure all that glitters is gold...", // Preview from matching corpus text
        "similarity_score": 1.0
      }
    ],
    "llm_risk_assessment": { // Populated if matches were found
      "overall_risk_level": "High",
      "assessment_notes": "Significant verbatim overlap detected between input and corpus snippet index 1."
    }
  }
}
```
*(Note: lyric_analysis will be null if no lyrics are provided. llm_risk_assessment will be null if no matches are found above the similarity threshold).*

## Evaluation Strategy

While this project doesn't include a pre-built labeled dataset for quantitative evaluation due to budget and time constraints, here's how the quality of the generated metadata could be assessed:

1. **Semantic Tagging (Genre, Mood, Instruments):**
   - **Metrics:** Precision, Recall, F1-Score (Micro and Macro averages), Hamming Loss. These are standard metrics for multi-label classification.
   - **Dataset:** A dataset like MTG-Jamendo (requires registration), which contains audio features and corresponding genre/mood tags assigned by humans, could be used as ground truth.
   - **Process:**
     - Run the audio files from the evaluation dataset through the analyzer and tagger modules.
     - Compare the LLM-generated tags (semantic_tags output) against the ground truth tags from the dataset.
     - Calculate Precision (accuracy of predicted tags), Recall (coverage of true tags), and F1-Score (harmonic mean) to assess tag quality.

2. **Descriptive Metadata (Track Description, SEO Keywords):**
   - **Metrics:** ROUGE (ROUGE-1, ROUGE-2, ROUGE-L), BLEU scores, or qualitative human assessment.
   - **Dataset:** Requires a dataset where tracks have human-written descriptions or curated keywords. Alternatively, human evaluation is needed.
   - **Process:**
     - Generate descriptions using the metadata module for tracks with reference descriptions.
     - Calculate ROUGE/BLEU scores comparing the generated description to the human reference. ROUGE measures recall-oriented n-gram overlap (good for summaries), while BLEU is precision-oriented (good for translation-like tasks).
     - For keywords, metrics like Precision/Recall/F1 could be used if reference keywords exist.
     - Alternatively, human evaluators could rate the generated descriptions/keywords based on relevance, coherence, and appeal.

Implementing these evaluations would provide quantitative insights into the model's performance and guide further prompt engineering or model selection.

## Limitations & Future Work

**Limitations:**
- **Local LLM Dependency:** Performance relies heavily on the local machine's ability to run Llama 3 via Ollama efficiently. Generation can be slow on less powerful hardware.
- **Rights Check Scope:** The current lyric similarity check is basic.
  - It compares the full input lyrics to full corpus documents. Implementing corpus chunking (splitting large corpus texts into smaller pieces for comparison) would be necessary for accurately finding excerpts within larger works.
  - The quality depends heavily on the size and relevance of the corpus/ directory contents.
  - The LLM risk assessment is experimental and based only on text similarity, not legal interpretation.
- **Prompt Sensitivity:** LLM outputs can vary and may require further prompt tuning for consistency and accuracy across diverse musical inputs.
- **No UI:** The current interfaces are CLI and a basic Flask API. A proper web front-end could be built.
- **Error Handling:** Can be made more robust, especially around LLM API interactions and file processing.
- **Scalability:** The current implementation processes tracks sequentially.

**Potential Future Enhancements:**
- Implement corpus chunking for the rights check.
- Add more sophisticated audio features (e.g., harmonic/percussive separation, detailed rhythm analysis).
- Develop a web-based user interface (e.g., using React/Vue/Svelte with the Flask API).
- Containerize the application using Docker for easier deployment.
- Experiment with different LLMs (local or API-based).
- Integrate evaluation metrics calculation using a suitable dataset.
- Add support for batch processing multiple files.