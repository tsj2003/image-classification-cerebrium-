# Image Classification API Deployment

This project implements an image classification API using a PyTorch model converted to ONNX format, deployed on Cerebrium's serverless GPU platform.

## Project Structure

```
.
├── src/
│   ├── app.py              # FastAPI application
│   │   └── model.py        # ONNX model implementation
│   ├── model/
│   │   └── pytorch_model.py # PyTorch model implementation
│   ├── tests/
│   │   ├── test_server.py  # Cerebrium deployment tests
│   │   └── test_model.py   # Model unit tests
│   └── utils/              # Utility functions
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── convert_to_onnx.py      # PyTorch to ONNX conversion script
└── README.md              # This file
```

## Prerequisites

- Python 3.10 or higher
- Docker
- Cerebrium account (with API key)
- Git

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download model weights:
   ```bash
   # Download PyTorch weights
   curl -L "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0" -o pytorch_model_weights.pth
   
   # Convert to ONNX
   python convert_to_onnx.py
   ```

## Local Development

1. Start the API server:
   ```bash
   MODEL_PATH=./model.onnx uvicorn src.app:app --reload --port 8000
   ```

2. Run tests:
   ```bash
   # Run model tests
   pytest src/tests/test_model.py
   
   # Run server tests
   python src/tests/test_server.py --api_key localtest --api_url http://127.0.0.1:8000 --run_tests
   ```

## Cerebrium Deployment

1. Build Docker image:
   ```bash
   docker build -t mtailor-image-classifier:latest .
   ```

2. Deploy to Cerebrium:
   - Log in to Cerebrium dashboard
   - Create new deployment
   - Select "Custom Docker Image"
   - Use the built image
   - Configure environment variables:
     - MODEL_PATH=/app/model.onnx
     - API_KEY=your_api_key

3. Test deployment:
   ```bash
   python src/tests/test_server.py --api_key your_cerebrium_api_key --api_url your_cerebrium_endpoint --run_tests
   ```

## API Usage

### Endpoints

- `POST /predict`
  - Input: Image file
  - Output: JSON with class ID and confidence score
  - Headers: 
    - Authorization: Bearer your_api_key
    - Content-Type: multipart/form-data

### Example Request

```bash
curl -X POST "http://your-endpoint/predict" \
     -H "Authorization: Bearer your_api_key" \
     -F "file=@path/to/image.jpg"
```

### Example Response

```json
{
    "class_id": 0,
    "confidence": 0.9137
}
```

## Testing

The project includes comprehensive tests:

1. Model Tests:
   - ONNX model loading
   - Image preprocessing
   - Prediction accuracy
   - Performance benchmarks

2. Server Tests:
   - API endpoint functionality
   - Authentication
   - Error handling
   - Response times
   - Cerebrium deployment monitoring

## CI/CD

GitHub Actions workflow automatically:
- Runs tests on push and pull requests
- Builds Docker image
- Caches Docker layers for faster builds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Contact

[Your contact information]

## Notes

- The model expects RGB images of size 224x224
- Preprocessing includes:
  - RGB conversion
  - Resize to 224x224 (bilinear interpolation)
  - Normalization using ImageNet mean and std
- Response time target: 2-3 seconds
- Free Cerebrium credits: 30 USD (sufficient for testing)

## Troubleshooting

1. Model conversion issues:
   - Ensure PyTorch version compatibility
   - Check model weights file integrity

2. Deployment issues:
   - Verify Docker image builds successfully
   - Check Cerebrium logs for errors
   - Ensure environment variables are set correctly

3. API issues:
   - Verify API key and endpoint URL
   - Check image format and size
   - Monitor response times


