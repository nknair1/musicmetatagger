# app/analyzer.py
from pathlib import Path
import librosa
import numpy as np
import logging # Use logging

# Configure logger for this module
logger = logging.getLogger(__name__)
# Set basic config if not already set by another module (e.g., main.py)
# This ensures logs appear even if analyzer is run standalone or imported first
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_features(audio_path: Path, sr: int = 22050) -> dict | None:
    """
    Extracts tempo (BPM), key, MFCCs, RMS energy, spectral centroid,
    and zero-crossing rate from an audio file. Returns None if loading fails.
    """
    try:
        logger.debug(f"Loading audio file: {audio_path} with sr={sr}")
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        logger.debug("Audio loaded successfully.")

        # --- Tempo ---
        # Use librosa.feature.tempo to get BPM estimate directly
        tempo_estimate = librosa.feature.tempo(y=y, sr=sr)
        # tempo usually returns an array, we typically take the first estimate
        tempo_bpm = float(np.round(tempo_estimate[0], 2)) if tempo_estimate.size > 0 else 0.0
        logger.debug(f"Estimated tempo: {tempo_bpm} BPM")

        # --- Key ---
        # Chroma features are used for key estimation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        # Simple key estimation: find the chroma bin with max energy
        key_idx = np.argmax(np.sum(chroma, axis=1))
        # Convert MIDI note number (starting from C=0) to note name
        # Librosa's note mapping might need adjustment depending on convention (C=0 or C=24)
        # midi_to_note assumes C0=0, A4=69. Let's map chroma index directly.
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key_note = note_names[key_idx % 12]
        # (This doesn't distinguish major/minor, just the root note)
        logger.debug(f"Estimated key root note: {estimated_key_note}")


        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        logger.debug("MFCCs calculated.")

        # --- RMS Energy ---
        rms = librosa.feature.rms(y=y)[0] # Get RMS energy per frame
        rms_mean = np.mean(rms)
        logger.debug(f"Mean RMS energy calculated: {rms_mean:.4f}")

        # --- Spectral Centroid ---
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        logger.debug(f"Mean Spectral Centroid calculated: {spectral_centroid_mean:.2f}")

        # --- Zero-Crossing Rate ---
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        logger.debug(f"Mean Zero-Crossing Rate calculated: {zcr_mean:.4f}")

        return {
            "tempo_bpm": tempo_bpm,
            "estimated_key_root": estimated_key_note, # Renamed for clarity
            "mfcc_mean": mfcc_mean.tolist(),
            "rms_mean": float(rms_mean),
            "spectral_centroid_mean": float(spectral_centroid_mean),
            "zcr_mean": float(zcr_mean)
        }
    except Exception as e:
        # Use the logger configured for this module
        logger.error(f"Error processing {audio_path.name}: {e}", exc_info=True) # Log full traceback
        return None