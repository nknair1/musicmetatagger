from pathlib import Path
from app.analyzer import extract_features

def test_tempo_key():
    feat = extract_features(Path(__file__).parent / "data" / "120bpm_D.wav")
    assert 117 <= feat["tempo_bpm"] <= 123
    assert feat["key"].startswith("D")
