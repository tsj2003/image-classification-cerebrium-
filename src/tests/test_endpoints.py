import os
import requests
import json
from datetime import datetime
from typing import Dict, Any
import base64
from PIL import Image
import io

class APITester:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.results = []
        
    def test_endpoint(self, method: str, endpoint: str, expected_status: int, 
                     data: Dict[str, Any] = None, headers: Dict[str, str] = None, files=None) -> Dict[str, Any]:
        """Test an endpoint and return the result"""
        url = f"{self.base_url}{endpoint}"
        try:
            if files:
                response = requests.request(method, url, data=data, headers=headers, files=files)
            else:
                response = requests.request(method, url, json=data, headers=headers)
            success = response.status_code == expected_status
            result = {
                "endpoint": endpoint,
                "method": method,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "success": success,
                "response": response.json() if response.headers.get('content-type') == 'application/json' else response.text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            result = {
                "endpoint": endpoint,
                "method": method,
                "expected_status": expected_status,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        self.results.append(result)
        return result

    def test_predict_with_image(self, image_path: str, expected_status: int = 200) -> Dict[str, Any]:
        """Test the predict endpoint with an image as file upload"""
        try:
            with open(image_path, 'rb') as img_file:
                files = {'file': img_file}
                headers = {"Authorization": f"Bearer {os.getenv('CEREBRIUM_API_KEY', 'test_api_key')}"}
                return self.test_endpoint("POST", "/predict", expected_status, headers=headers, files=files)
        except Exception as e:
            result = {
                "endpoint": "/predict",
                "method": "POST",
                "expected_status": expected_status,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
            return result

    def run_all_tests(self, test_image_path: str = "test_images/test_image.jpg") -> None:
        """Run all endpoint tests"""
        print("\nRunning API Endpoint Tests...")
        print("-" * 50)

        # Test root endpoint
        self.test_endpoint("GET", "/", 200)
        
        # Test health endpoint
        self.test_endpoint("GET", "/health", 200)
        
        # Test metrics endpoint
        self.test_endpoint("GET", "/metrics", 200)
        
        # Test predict endpoint without auth
        self.test_endpoint("POST", "/predict", 401, 
                          data={"image": "dummy"})
        
        # Test predict endpoint with invalid image
        self.test_endpoint("POST", "/predict", 400,
                          data={"image": "invalid"},
                          headers={"Authorization": f"Bearer {os.getenv('CEREBRIUM_API_KEY', 'test_api_key')}"})
        
        # Test predict endpoint with valid image
        self.test_predict_with_image(test_image_path)

    def print_results(self) -> None:
        """Print test results in a formatted way"""
        print("\nTest Results Summary:")
        print("-" * 50)
        
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['method']} {result['endpoint']}")
            print(f"   Expected Status: {result['expected_status']}")
            print(f"   Actual Status: {result.get('actual_status', 'N/A')}")
            if "error" in result:
                print(f"   Error: {result['error']}")
            if "response" in result:
                print(f"   Response: {json.dumps(result['response'], indent=2)}")
            print("-" * 50)

def main():
    # Create test directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Create a test image if it doesn't exist
    test_image_path = "test_images/test_image.jpg"
    if not os.path.exists(test_image_path):
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(test_image_path)
    
    # Run tests
    tester = APITester()
    tester.run_all_tests(test_image_path)
    tester.print_results()

if __name__ == "__main__":
    main() 