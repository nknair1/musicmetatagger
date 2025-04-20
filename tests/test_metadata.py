# tests/test_metadata.py
import pytest
import json
from unittest.mock import MagicMock, patch # Import patch if mocking client directly
from app import metadata # Import the module to test

# Define sample tags input
SAMPLE_TAGS = {
    "primary_genre": "Electronic",
    "sub_genres": ["Synthwave"],
    "mood_tags": ["Nostalgic", "Driving"],
    "instruments_likely": ["Synthesizer", "Drum Machine"]
}

# Define expected successful JSON output (as a string, like Ollama might return)
MOCK_OLLAMA_METADATA_RESPONSE = json.dumps({
  "track_description": "A driving Synthwave track with nostalgic synthesizers.",
  "seo_keywords": ["synthwave", "electronic", "nostalgic", "driving", "synthesizer"]
})

# --- Test Cases ---

# Use pytest-mock's 'mocker' fixture for easier mocking within tests
def test_build_track_metadata_success(mocker):
    """Test successful metadata generation with valid tags and mocked Ollama."""
    # Mock the 'client' object within the metadata module
    mock_client = MagicMock()
    # Configure the mock 'generate' method to return the expected structure
    mock_client.generate.return_value = {'response': MOCK_OLLAMA_METADATA_RESPONSE}

    # Patch the 'client' instance *within the metadata module* for this test
    mocker.patch('app.metadata.client', mock_client)

    result = metadata.build_track_metadata(SAMPLE_TAGS)

    # Assertions
    mock_client.generate.assert_called_once() # Check Ollama was called
    assert result is not None
    assert "track_description" in result
    assert "seo_keywords" in result
    assert result["track_description"] == "A driving Synthwave track with nostalgic synthesizers."
    assert "synthwave" in result["seo_keywords"]

def test_build_track_metadata_ollama_json_error(mocker):
    """Test handling of invalid JSON response from Ollama."""
    mock_client = MagicMock()
    # Simulate Ollama returning malformed JSON
    mock_client.generate.return_value = {'response': '{"bad json"'}
    mocker.patch('app.metadata.client', mock_client)

    result = metadata.build_track_metadata(SAMPLE_TAGS)

    mock_client.generate.assert_called_once()
    assert result is None # Expect None based on error handling

def test_build_track_metadata_ollama_api_error(mocker):
    """Test handling of exception during Ollama API call."""
    mock_client = MagicMock()
    # Simulate an exception during the generate call
    mock_client.generate.side_effect = Exception("Ollama connection failed")
    mocker.patch('app.metadata.client', mock_client)

    result = metadata.build_track_metadata(SAMPLE_TAGS)

    mock_client.generate.assert_called_once()
    assert result is None

def test_build_track_metadata_no_client(mocker):
    """Test behavior when Ollama client is None (failed initialization)."""
    # Patch the client to be None for this test
    mocker.patch('app.metadata.client', None)

    result = metadata.build_track_metadata(SAMPLE_TAGS)
    assert result is None

def test_build_track_metadata_empty_tags():
    """Test providing empty or invalid tags."""
    result_none = metadata.build_track_metadata(None)
    result_empty = metadata.build_track_metadata({})
    assert result_none is None
    assert result_empty is None
    # Check that Ollama wasn't called (optional, depends if you patched client)

def test_build_track_metadata_missing_keys(mocker):
    """Test handling when Ollama returns JSON missing required keys."""
    mock_client = MagicMock()
    # Simulate Ollama returning valid JSON but missing 'seo_keywords'
    mock_response_missing_keys = json.dumps({
        "track_description": "A description."
        # Missing seo_keywords
    })
    mock_client.generate.return_value = {'response': mock_response_missing_keys}
    mocker.patch('app.metadata.client', mock_client)

    result = metadata.build_track_metadata(SAMPLE_TAGS)

    mock_client.generate.assert_called_once()
    assert result is None # Based on current validation logic in metadata.py