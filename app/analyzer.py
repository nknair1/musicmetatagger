from pathlib import Path
import librosa
import numpy as np

def extract_features(audio_path: Path, sr: int = 22050) -> dict | None:
    """
    Extracts tempo, key, MFCCs, RMS energy, and spectral centroid
    from an audio file. Returns None if loading fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        # --- Existing Features ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, units='bpm') # Get tempo directly in BPM
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        # Estimate key more robustly (optional, basic chroma max is okay too)
        key_corr = np.corrcoef(chroma, librosa.key_to_chroma(key='C:maj')) # Example correlation
        key_idx = np.argmax(np.sum(chroma, axis=1)) # Simplified back to max chroma bin
        estimated_key = librosa. midi_to_note(key_idx % 12 + 24) # Get note name

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)

        # --- New Features ---
        rms = librosa.feature.rms(y=y)[0] # Get RMS energy per frame, take the first element (often index [0])
        rms_mean = np.mean(rms)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)

        # Maybe add Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)

        return {
            "tempo_bpm": float(np.round(tempo[0], 2)) if tempo.size > 0 else 0.0, # Handle case where tempo detection might fail
            "estimated_key": estimated_key,
            "mfcc_mean": mfcc_mean.tolist(), # Average MFCCs over time
            "rms_mean": float(rms_mean),
            "spectral_centroid_mean": float(spectral_centroid_mean),
            "zcr_mean": float(zcr_mean)
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Consider using proper logging here instead of print
        import logging
        logging.error(f"Error processing {audio_path}", exc_info=True)
        return None