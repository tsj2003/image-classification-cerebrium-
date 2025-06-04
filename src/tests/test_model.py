import os
import pytest
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.model import ONNXModel, ImagePreprocessor
from src.utils.convert_to_onnx import convert_to_onnx

# Test data paths
TEST_IMAGES = {
    'tench': 'n01440764_tench.jpeg',
    'mud_turtle': 'n01667114_mud_turtle.JPEG'
}

@pytest.fixture
def model_path(tmp_path):
    """Fixture to create a temporary ONNX model for testing."""
    # Path to PyTorch weights (you'll need to download this)
    pytorch_weights = "pytorch_model_weights.pth"
    onnx_path = tmp_path / "test_model.onnx"
    
    # Convert model to ONNX
    convert_to_onnx(
        model_path=pytorch_weights,
        output_path=str(onnx_path),
        opset_version=12
    )
    
    return str(onnx_path)

@pytest.fixture
def model(model_path):
    """Fixture to create a model instance for testing."""
    return ONNXModel(model_path)

@pytest.fixture
def preprocessor():
    """Fixture to create a preprocessor instance for testing."""
    return ImagePreprocessor()

def test_preprocessor_output_shape(preprocessor):
    """Test that preprocessor outputs correct shape."""
    for image_name in TEST_IMAGES.values():
        output = preprocessor.preprocess(image_name)
        assert output.shape == (1, 3, 224, 224)
        assert output.dtype == np.float32

def test_preprocessor_normalization(preprocessor):
    """Test that preprocessor normalizes images correctly."""
    for image_name in TEST_IMAGES.values():
        output = preprocessor.preprocess(image_name)
        # Check that values are roughly normalized
        assert np.all(output >= -3) and np.all(output <= 3)

def test_model_initialization(model_path):
    """Test model initialization."""
    # Test successful initialization
    model = ONNXModel(model_path)
    assert model is not None
    
    # Test initialization with non-existent model
    with pytest.raises(FileNotFoundError):
        ONNXModel("non_existent_model.onnx")

def test_model_prediction(model):
    """Test model prediction on test images."""
    # Test prediction on tench image
    class_id, confidence = model.predict(TEST_IMAGES['tench'])
    assert class_id == 0  # tench class ID
    assert 0 <= confidence <= 1
    
    # Test prediction on mud turtle image
    class_id, confidence = model.predict(TEST_IMAGES['mud_turtle'])
    assert class_id == 35  # mud turtle class ID
    assert 0 <= confidence <= 1

def test_model_batch_prediction(model):
    """Test batch prediction functionality."""
    image_paths = list(TEST_IMAGES.values())
    predictions = model.predict_batch(image_paths)
    
    assert len(predictions) == len(image_paths)
    for class_id, confidence in predictions:
        assert isinstance(class_id, int)
        assert 0 <= confidence <= 1

def test_model_invalid_input(model):
    """Test model behavior with invalid inputs."""
    # Test with non-existent image
    with pytest.raises(FileNotFoundError):
        model.predict("non_existent_image.jpg")
    
    # Test with invalid image format
    with pytest.raises(Exception):
        model.predict("requirements.txt")  # Using a text file as image

def test_model_performance(model):
    """Test model performance (response time)."""
    import time
    
    start_time = time.time()
    model.predict(TEST_IMAGES['tench'])
    end_time = time.time()
    
    # Check if inference time is within acceptable range (2-3 seconds)
    inference_time = end_time - start_time
    assert inference_time <= 3.0, f"Inference took {inference_time:.2f} seconds, which is too slow" 