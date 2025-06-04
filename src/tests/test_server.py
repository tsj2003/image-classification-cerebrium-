import os
import sys
import argparse
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TestResult:
    """Class to store test results"""
    image_path: str
    expected_class: int
    predicted_class: int
    confidence: float
    response_time: float
    success: bool
    error: Optional[str] = None

class CerebriumTester:
    """Class to test the deployed model on Cerebrium."""
    
    def __init__(self, api_key: str, api_url: str):
        """
        Initialize the tester.
        
        Args:
            api_key (str): Cerebrium API key
            api_url (str): Cerebrium API endpoint URL
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.test_results: List[TestResult] = []
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "errors": {}
        }
    
    def predict(self, image_path: str) -> Tuple[Dict, float]:
        """
        Make a prediction request to the deployed model
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Tuple[Dict, float]: Prediction response and response time
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Make request
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=self.headers,
                files={'file': (os.path.basename(image_path), image_data)}
            )
            response_time = time.time() - start_time
            
            # Update metrics
            self.metrics["total_requests"] += 1
            self.metrics["total_response_time"] += response_time
            self.metrics["min_response_time"] = min(self.metrics["min_response_time"], response_time)
            self.metrics["max_response_time"] = max(self.metrics["max_response_time"], response_time)
            
            if response.status_code == 200:
                self.metrics["successful_requests"] += 1
                return response.json(), response_time
            else:
                self.metrics["failed_requests"] += 1
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                self.metrics["errors"][error_msg] = self.metrics["errors"].get(error_msg, 0) + 1
                raise Exception(error_msg)
                
        except Exception as e:
            self.metrics["failed_requests"] += 1
            error_msg = str(e)
            self.metrics["errors"][error_msg] = self.metrics["errors"].get(error_msg, 0) + 1
            raise
    
    def run_preset_tests(self, test_images: List[Tuple[str, int]]) -> List[TestResult]:
        """
        Run a set of preset tests with known expected classes
        
        Args:
            test_images (List[Tuple[str, int]]): List of (image_path, expected_class) tuples
            
        Returns:
            List[TestResult]: List of test results
        """
        print("\nRunning preset tests...")
        for image_path, expected_class in test_images:
            try:
                print(f"\nTesting image: {image_path}")
                print(f"Expected class: {expected_class}")
                
                # Make prediction
                response, response_time = self.predict(image_path)
                predicted_class = response["class_id"]
                confidence = response["confidence"]
                
                # Create test result
                result = TestResult(
                    image_path=image_path,
                    expected_class=expected_class,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    response_time=response_time,
                    success=(predicted_class == expected_class)
                )
                
                # Print result
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Response time: {response_time:.3f}s")
                print(f"Test {'passed' if result.success else 'failed'}")
                
                self.test_results.append(result)
                
            except Exception as e:
                print(f"Error testing {image_path}: {str(e)}")
                self.test_results.append(TestResult(
                    image_path=image_path,
                    expected_class=expected_class,
                    predicted_class=-1,
                    confidence=0.0,
                    response_time=0.0,
                    success=False,
                    error=str(e)
                ))
        
        return self.test_results
    
    def generate_report(self, output_dir: str = "test_reports"):
        """
        Generate a test report with metrics and visualizations
        
        Args:
            output_dir (str): Directory to save the report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        avg_response_time = self.metrics["total_response_time"] / self.metrics["total_requests"] if self.metrics["total_requests"] > 0 else 0
        
        # Generate report
        report = {
            "timestamp": timestamp,
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "avg_response_time": avg_response_time,
                "min_response_time": self.metrics["min_response_time"],
                "max_response_time": self.metrics["max_response_time"],
                "total_requests": self.metrics["total_requests"],
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"]
            },
            "errors": self.metrics["errors"],
            "test_results": [
                {
                    "image_path": r.image_path,
                    "expected_class": r.expected_class,
                    "predicted_class": r.predicted_class,
                    "confidence": r.confidence,
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error
                }
                for r in self.test_results
            ]
        }
        
        # Save report
        report_path = os.path.join(output_dir, f"test_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(output_dir, timestamp)
        
        print(f"\nTest report generated: {report_path}")
        return report_path
    
    def _generate_visualizations(self, output_dir: str, timestamp: str):
        """Generate visualization plots for the test results"""
        # Response time distribution
        response_times = [r.response_time for r in self.test_results if r.success]
        if response_times:
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=20)
            plt.title("Response Time Distribution")
            plt.xlabel("Response Time (s)")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, f"response_times_{timestamp}.png"))
            plt.close()
        
        # Success rate by image
        if self.test_results:
            plt.figure(figsize=(12, 6))
            image_names = [os.path.basename(r.image_path) for r in self.test_results]
            success_rates = [1 if r.success else 0 for r in self.test_results]
            plt.bar(image_names, success_rates)
            plt.title("Test Success Rate by Image")
            plt.xlabel("Image")
            plt.ylabel("Success (1) / Failure (0)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"success_rates_{timestamp}.png"))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test deployed model on Cerebrium')
    parser.add_argument('--api_key', type=str, required=True,
                      help='Cerebrium API key')
    parser.add_argument('--api_url', type=str, required=True,
                      help='Cerebrium API endpoint URL')
    parser.add_argument('--test_images', type=str, nargs='+',
                      help='Paths to test images')
    parser.add_argument('--expected_classes', type=int, nargs='+',
                      help='Expected class IDs for test images')
    parser.add_argument('--output_dir', type=str, default='test_reports',
                      help='Directory to save test reports')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_images and args.expected_classes:
        if len(args.test_images) != len(args.expected_classes):
            raise ValueError("Number of test images must match number of expected classes")
        test_images = list(zip(args.test_images, args.expected_classes))
    else:
        # Use default test images
        test_images = [
            ("n01440764_tench.jpeg", 0),
            ("n01667114_mud_turtle.JPEG", 35)
        ]
    
    # Initialize tester
    tester = CerebriumTester(args.api_key, args.api_url)
    
    try:
        # Run tests
        results = tester.run_preset_tests(test_images)
        
        # Generate report
        report_path = tester.generate_report(args.output_dir)
        
        # Print summary
        print("\nTest Summary:")
        print(f"Total tests: {len(results)}")
        print(f"Successful tests: {sum(1 for r in results if r.success)}")
        print(f"Failed tests: {sum(1 for r in results if not r.success)}")
        print(f"Average response time: {tester.metrics['total_response_time'] / tester.metrics['total_requests']:.3f}s")
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        raise

if __name__ == '__main__':
    main() 