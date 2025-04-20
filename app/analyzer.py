from pathlib import Path
import librosa
import numpy as np

def extract_features(audio_path: Path, sr: int = 22050) -> dict:
    """Return tempo, key, and 13‑dim MFCC vector from an audio file."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = chroma.mean(axis=1).argmax()
    key = librosa.hz_to_note(librosa.midi_to_hz(key_idx + 24))  # rough
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    return {
        "tempo_bpm": float(np.round(tempo, 2)),
        "key": key,
        "mfcc": mfcc.tolist(),
    }
