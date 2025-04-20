import pytest
from unittest.mock import patch, MagicMock
import json
from app.tagger import tag_track

# Example features to use in tests
DUMMY_FEATURES = {"tempo_bpm": 120.0, "key": "Cmaj", "mfcc": [0.0]*13}

# Expected successful JSON output (as a string, like Ollama might return)
MOCK_OLLAMA_SUCCESS_RESPONSE = json.dumps({
    "primary_genre": "Test Genre",
    "sub_genres": ["Test Sub"],
    "mood_tags": ["Test Mood"],
    "instruments_likely": ["Test Instrument"]
})

@patch('app.tagger.client') # Mock the 'client' object within app.tagger
def test_tag_track_success(mock_ollama_client):
    """Test tag_track successfully parses a good JSON response from mocked Ollama."""
    # Configure the mock client's generate method
    mock_response = {'response': MOCK_OLLAMA_SUCCESS_RESPONSE}
    mock_ollama_client.generate.return_value = mock_response

    # Call the function with dummy features
    result = tag_track(DUMMY_FEATURES)

    # Assertions
    mock_ollama_client.generate.assert_called_once() # Check if generate was called
    assert result is not None
    assert result["primary_genre"] == "Test Genre"
    assert "Test Sub" in result["sub_genres"]

@patch('app.tagger.client')
def test_tag_track_json_decode_error(mock_ollama_client):
    """Test tag_track handles invalid JSON response gracefully."""
    # Configure mock to return bad JSON
    mock_response = {'response': '{"bad json"'} # Malformed JSON string
    mock_ollama_client.generate.return_value = mock_response

    result = tag_track(DUMMY_FEATURES)

    # Assertions
    mock_ollama_client.generate.assert_called_once()
    assert result is None # Expect None on decode error based on your function's logic

@patch('app.tagger.client', None) # Mock client to be None, simulating connection failure
def test_tag_track_no_client():
    """Test tag_track handles case where Ollama client is None."""
    result = tag_track(DUMMY_FEATURES)
    assert result is None

