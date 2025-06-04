import os
os.environ["CEREBRIUM_API_KEY"] = "test_api_key"
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import time
from src.app import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Fixture to provide a test image path."""
    return "n01440764_tench.jpeg"

@pytest.fixture
def api_key():
    """Fixture to provide a test API key."""
    return "test_api_key"

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "Image Classification API is running"
    assert data["version"] == "1.0.0"
    assert data["platform"] == "Cerebrium"

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert isinstance(data["model_loaded"], bool)
    assert "api_stats" in data
    assert "environment" in data
    assert data["environment"]["model_path"] == os.getenv("CEREBRIUM_MODEL_PATH", "model.onnx")

def test_metrics_endpoint(client):
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "api_stats" in data
    assert "model_info" in data
    assert isinstance(data["api_stats"]["total_requests"], int)
    assert isinstance(data["api_stats"]["successful_requests"], int)
    assert isinstance(data["api_stats"]["failed_requests"], int)

def test_predict_endpoint_without_auth(client, test_image):
    """Test the predict endpoint without authentication."""
    # Set API key in environment
    os.environ["CEREBRIUM_API_KEY"] = "test_api_key"
    
    with open(test_image, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpeg", f, "image/jpeg")}
        )
    
    # Should fail without auth header
    assert response.status_code == 401
    assert "API key is required" in response.json()["detail"]

def test_predict_endpoint_with_auth(client, test_image, api_key):
    """Test the predict endpoint with authentication."""
    # Set API key in environment
    os.environ["CEREBRIUM_API_KEY"] = api_key
    
    with open(test_image, "rb") as f:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("test_image.jpeg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "confidence" in data
    assert "filename" in data
    assert "processing_time" in data
    assert isinstance(data["class_id"], int)
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1

def test_predict_endpoint_invalid_image(client, api_key):
    """Test the predict endpoint with an invalid image."""
    os.environ["CEREBRIUM_API_KEY"] = api_key
    
    # Create an invalid image file
    with open("invalid.txt", "w") as f:
        f.write("This is not an image")
    
    with open("invalid.txt", "rb") as f:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    assert "Error processing image" in response.json()["detail"]
    
    # Clean up
    os.remove("invalid.txt")

def test_metrics_tracking(client, test_image, api_key):
    """Test that metrics are properly tracked across requests."""
    os.environ["CEREBRIUM_API_KEY"] = api_key
    
    # Get initial metrics
    initial_metrics = client.get("/metrics").json()
    initial_total = initial_metrics["api_stats"]["total_requests"]
    initial_success = initial_metrics["api_stats"]["successful_requests"]
    
    # Make a successful prediction
    with open(test_image, "rb") as f:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("test_image.jpeg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    
    # Get updated metrics
    updated_metrics = client.get("/metrics").json()
    assert updated_metrics["api_stats"]["total_requests"] == initial_total + 1
    assert updated_metrics["api_stats"]["successful_requests"] == initial_success + 1
    
    # Make a failed request
    with open("requirements.txt", "rb") as f:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("requirements.txt", f, "text/plain")}
        )
    assert response.status_code == 400
    
    # Check final metrics
    final_metrics = client.get("/metrics").json()
    assert final_metrics["api_stats"]["total_requests"] == initial_total + 2
    assert final_metrics["api_stats"]["failed_requests"] > initial_metrics["api_stats"]["failed_requests"]

def test_response_time_tracking(client, test_image, api_key):
    """Test that response times are properly tracked."""
    os.environ["CEREBRIUM_API_KEY"] = api_key
    
    # Make a prediction and verify processing time
    with open(test_image, "rb") as f:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("test_image.jpeg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "processing_time" in data
    assert isinstance(data["processing_time"], float)
    assert data["processing_time"] > 0
    
    # Check that average response time is updated in metrics
    metrics = client.get("/metrics").json()
    assert "average_response_time" in metrics["api_stats"]
    assert metrics["api_stats"]["average_response_time"] > 0 