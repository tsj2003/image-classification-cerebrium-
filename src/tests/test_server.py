import os
import sys
import argparse
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

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
    
    def predict(self, image_path: str) -> Tuple[int, float]:
        """
        Make a prediction using the deployed model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[int, float]: Predicted class ID and confidence score
        """
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare request
        files = {'file': (os.path.basename(image_path), image_data)}
        
        # Make request to /predict endpoint
        predict_url = f"{self.api_url.rstrip('/')}/predict"
        start_time = time.time()
        response = requests.post(
            predict_url,
            headers={'Authorization': f'Bearer {self.api_key}'},
            files=files
        )
        end_time = time.time()
        
        # Check response
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        # Parse response
        result = response.json()
        prediction_time = end_time - start_time
        
        return result['class_id'], result['confidence'], prediction_time
    
    def run_preset_tests(self) -> Dict:
        """
        Run a set of predefined tests on the deployed model.
        
        Returns:
            Dict: Test results including accuracy and performance metrics
        """
        test_images = {
            'tench': 'n01440764_tench.jpeg',
            'mud_turtle': 'n01667114_mud_turtle.JPEG'
        }
        
        expected_classes = {
            'tench': 0,
            'mud_turtle': 35
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_images),
            'passed_tests': 0,
            'failed_tests': 0,
            'average_response_time': 0,
            'test_details': []
        }
        
        total_response_time = 0
        
        for image_name, image_path in test_images.items():
            try:
                class_id, confidence, response_time = self.predict(image_path)
                total_response_time += response_time
                
                expected_class = expected_classes[image_name]
                test_passed = class_id == expected_class
                
                test_result = {
                    'image': image_name,
                    'expected_class': expected_class,
                    'predicted_class': class_id,
                    'confidence': confidence,
                    'response_time': response_time,
                    'passed': test_passed
                }
                
                if test_passed:
                    results['passed_tests'] += 1
                else:
                    results['failed_tests'] += 1
                
                results['test_details'].append(test_result)
                
            except Exception as e:
                results['failed_tests'] += 1
                results['test_details'].append({
                    'image': image_name,
                    'error': str(e),
                    'passed': False
                })
        
        results['average_response_time'] = total_response_time / len(test_images)
        return results
    
    def monitor_deployment(self, duration: int = 60, interval: int = 10) -> List[Dict]:
        """
        Monitor the deployed model's performance over time.
        
        Args:
            duration (int): Monitoring duration in seconds
            interval (int): Time between checks in seconds
            
        Returns:
            List[Dict]: List of monitoring results
        """
        monitoring_results = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Test with a known image
                class_id, confidence, response_time = self.predict('n01440764_tench.jpeg')
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy',
                    'response_time': response_time,
                    'class_id': class_id,
                    'confidence': confidence
                }
                
            except Exception as e:
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e)
                }
            
            monitoring_results.append(result)
            time.sleep(interval)
        
        return monitoring_results

def main():
    parser = argparse.ArgumentParser(description='Test deployed model on Cerebrium')
    parser.add_argument('--api_key', type=str, required=True,
                      help='Cerebrium API key')
    parser.add_argument('--api_url', type=str, required=True,
                      help='Cerebrium API endpoint URL')
    parser.add_argument('--image_path', type=str,
                      help='Path to image for single prediction')
    parser.add_argument('--run_tests', action='store_true',
                      help='Run preset tests')
    parser.add_argument('--monitor', action='store_true',
                      help='Monitor deployment')
    parser.add_argument('--monitor_duration', type=int, default=60,
                      help='Monitoring duration in seconds')
    parser.add_argument('--monitor_interval', type=int, default=10,
                      help='Time between monitoring checks in seconds')
    
    args = parser.parse_args()
    
    tester = CerebriumTester(args.api_key, args.api_url)
    
    if args.image_path:
        try:
            class_id, confidence, response_time = tester.predict(args.image_path)
            print(f"Prediction: Class {class_id} with confidence {confidence:.4f}")
            print(f"Response time: {response_time:.2f} seconds")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    if args.run_tests:
        print("\nRunning preset tests...")
        results = tester.run_preset_tests()
        print(f"\nTest Results:")
        print(f"Total tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Average response time: {results['average_response_time']:.2f} seconds")
        print("\nDetailed results:")
        for test in results['test_details']:
            print(f"\nImage: {test['image']}")
            if 'error' in test:
                print(f"Error: {test['error']}")
            else:
                print(f"Expected class: {test['expected_class']}")
                print(f"Predicted class: {test['predicted_class']}")
                print(f"Confidence: {test['confidence']:.4f}")
                print(f"Response time: {test['response_time']:.2f} seconds")
            print(f"Test passed: {test['passed']}")
    
    if args.monitor:
        print(f"\nMonitoring deployment for {args.monitor_duration} seconds...")
        results = tester.monitor_deployment(
            duration=args.monitor_duration,
            interval=args.monitor_interval
        )
        print("\nMonitoring Results:")
        for result in results:
            print(f"\nTimestamp: {result['timestamp']}")
            print(f"Status: {result['status']}")
            if result['status'] == 'healthy':
                print(f"Response time: {result['response_time']:.2f} seconds")
                print(f"Class ID: {result['class_id']}")
                print(f"Confidence: {result['confidence']:.4f}")
            else:
                print(f"Error: {result['error']}")

if __name__ == '__main__':
    main() 