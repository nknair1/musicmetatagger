# tests/test_rights.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from app import rights # Import the module to test

# --- Test Data ---
TEST_LYRICS = "These are the lyrics to test."
CORPUS_TEXTS = [
    "This is the first unrelated corpus text.",
    "This text contains the test lyrics fragment.", # Simulate potential match
    "Another unrelated document."
]
# Dummy embeddings (replace with actual expected shape if needed)
# Shape is typically (num_sentences, embedding_dim) - here simplified
DUMMY_EMBEDDING_INPUT = np.array([[0.1, 0.2, 0.3]])
DUMMY_EMBEDDINGS_CORPUS = np.array([
    [0.9, 0.8, 0.7], # Unrelated
    [0.11, 0.21, 0.31], # Similar to input
    [0.5, 0.5, 0.5] # Unrelated
])

# --- Mock SentenceTransformer ---
# Create a mock model instance outside tests if used repeatedly
mock_st_model = MagicMock()
# Configure the mock 'encode' method
# Use side_effect to return different values based on input
def encode_side_effect(texts, convert_to_tensor=False):
    # Note: In real usage, SentenceTransformer returns PyTorch tensors.
    # For mocking simplicity here, we use NumPy arrays directly
    # Adjust if your code relies heavily on tensor properties.
    if isinstance(texts, str) and texts == TEST_LYRICS:
        return DUMMY_EMBEDDING_INPUT # Return CPU array directly
    elif isinstance(texts, list) and texts == CORPUS_TEXTS:
        return DUMMY_EMBEDDINGS_CORPUS # Return CPU array directly
    else:
        # Handle unexpected calls if necessary
        raise ValueError("Mock encode called with unexpected input")

mock_st_model.encode.side_effect = encode_side_effect

# --- Test Cases ---

@pytest.fixture # Fixture to automatically patch the model for tests in this module
def patch_st_model(mocker):
    # Patch the 'model' instance *within the rights module*
    mocker.patch('app.rights.model', mock_st_model)
    # Also mock util.cos_sim if needed, or calculate manually for test
    # Mocking cos_sim directly gives more control for specific score testing
    mock_cos_sim = mocker.patch('app.rights.util.cos_sim')
    yield mock_cos_sim # Provide the mock cos_sim to tests if they need it

def test_check_lyric_similarity_match_found(patch_st_model): # Use the fixture
    """Test finding a similarity above the threshold."""
    mock_cos_sim = patch_st_model # Get the mock for util.cos_sim

    # --- Revised Mocking Strategy ---
    # 1. Create the mock tensor object that has the .cpu() method configured
    mock_tensor = MagicMock(name="MockTensor")
    mock_tensor.cpu.return_value = np.array([0.2, 0.85, 0.3]) # High score at index 1

    # 2. Make util.cos_sim return a mock sequence (list) containing mock_tensor
    # This simulates the behavior of tensor slicing/indexing [0]
    mock_cos_sim.return_value = [mock_tensor]
    # --- End Revised Mocking Strategy ---

    mock_st_model.encode.reset_mock()
    result = rights.check_lyric_similarity(TEST_LYRICS, CORPUS_TEXTS)

    # Assertions
    assert mock_st_model.encode.call_count == 2
    mock_cos_sim.assert_called_once()
    # Now assert that .cpu() was called on the mock_tensor we created
    mock_tensor.cpu.assert_called_once()
    assert len(result) == 1
    assert result[0]["corpus_index"] == 1
    assert result[0]["similarity_score"] == pytest.approx(0.85)
    assert result[0]["corpus_text_preview"].startswith("This text contains")

def test_check_lyric_similarity_no_match(patch_st_model): # Use the fixture
    """Test when no similarities are above the threshold."""
    mock_cos_sim = patch_st_model

    # --- Revised Mocking Strategy ---
    mock_tensor = MagicMock(name="MockTensor")
    # All scores <= 0.80
    mock_tensor.cpu.return_value = np.array([0.2, 0.75, 0.3])
    mock_cos_sim.return_value = [mock_tensor]
    # --- End Revised Mocking Strategy ---

    mock_st_model.encode.reset_mock()
    result = rights.check_lyric_similarity(TEST_LYRICS, CORPUS_TEXTS)

    assert mock_st_model.encode.call_count == 2
    mock_cos_sim.assert_called_once()
    mock_tensor.cpu.assert_called_once()
    assert len(result) == 0 # No matches expected

def test_check_lyric_similarity_empty_lyrics(patch_st_model):
    """Test providing empty input lyrics."""
    mock_cos_sim = patch_st_model
    mock_st_model.encode.reset_mock()

    result = rights.check_lyric_similarity("", CORPUS_TEXTS)

    assert result == []
    mock_st_model.encode.assert_not_called() # Should not attempt to encode
    mock_cos_sim.assert_not_called()

def test_check_lyric_similarity_empty_corpus(patch_st_model):
    """Test providing an empty corpus."""
    mock_cos_sim = patch_st_model
    mock_st_model.encode.reset_mock()

    result = rights.check_lyric_similarity(TEST_LYRICS, [])

    assert result == []
    # Encode might be called for input lyrics, but not for corpus
    # mock_st_model.encode.assert_called_once() # Check input encoding still happens
    mock_cos_sim.assert_not_called()

# You could add a test for when rights.model is None if desired
# def test_check_lyric_similarity_no_model(mocker): ...

# Test the load_corpus function (doesn't need mocking usually)
def test_load_corpus_success(tmp_path): # Use pytest's tmp_path fixture
    """Test loading text files from a directory."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    file1 = corpus_dir / "file1.txt"
    file1.write_text("Content of file 1.", encoding='utf-8')
    file2 = corpus_dir / "file2.txt"
    file2.write_text("Content of file 2.", encoding='utf-8')
    # Add an empty file
    empty_file = corpus_dir / "empty.txt"
    empty_file.touch()
    # Add a non-txt file
    other_file = corpus_dir / "image.jpg"
    other_file.touch()

    loaded_texts = rights.load_corpus(corpus_dir)

    assert len(loaded_texts) == 2
    assert "Content of file 1." in loaded_texts
    assert "Content of file 2." in loaded_texts

def test_load_corpus_dir_not_found():
    """Test when the corpus directory doesn't exist."""
    invalid_dir = Path("./non_existent_corpus_dir")
    loaded_texts = rights.load_corpus(invalid_dir)
    assert loaded_texts == []