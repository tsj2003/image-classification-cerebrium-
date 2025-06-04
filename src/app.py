import os
import time
from typing import Dict, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from datetime import datetime

from src.model.model import ONNXModel

# Initialize FastAPI app with Cerebrium-specific metadata
app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using ONNX model on Cerebrium platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Cerebrium-specific environment variables
API_KEY = os.getenv("CEREBRIUM_API_KEY", "")
MODEL_PATH = os.getenv("CEREBRIUM_MODEL_PATH", "model.onnx")
WORKERS = int(os.getenv("CEREBRIUM_WORKERS", "1"))
TIMEOUT = int(os.getenv("CEREBRIUM_TIMEOUT", "60"))

# Initialize model
model = ONNXModel(MODEL_PATH)

# Track API usage
api_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0,
    "last_request_time": None
}

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from header."""
    if not API_KEY:
        return True  # Skip verification if no API key is set
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )
    
    try:
        scheme, key = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication scheme"
            )
        if key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )
    
    return True

@app.get("/health")
async def health_check():
    """Detailed health check endpoint for Cerebrium."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "api_stats": api_stats,
        "environment": {
            "workers": WORKERS,
            "timeout": TIMEOUT,
            "model_path": MODEL_PATH
        }
    }

@app.get("/")
async def root():
    """Root endpoint with basic health check."""
    return {
        "status": "healthy",
        "message": "Image Classification API is running",
        "version": "1.0.0",
        "platform": "Cerebrium"
    }

@app.get("/metrics")
async def metrics():
    """Endpoint for monitoring metrics."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "api_stats": api_stats,
        "model_info": {
            "path": MODEL_PATH,
            "loaded": model is not None
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
) -> Dict:
    """
    Predict the class of an uploaded image.
    
    Args:
        file (UploadFile): The image file to classify
        
    Returns:
        Dict: Prediction results including class ID and confidence
    """
    start_time = time.time()
    api_stats["total_requests"] += 1
    api_stats["last_request_time"] = datetime.utcnow().isoformat()
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporarily for preprocessing
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)
        
        try:
            # Make prediction
            class_id, confidence = model.predict(temp_path)
            
            # Update stats
            api_stats["successful_requests"] += 1
            response_time = time.time() - start_time
            api_stats["average_response_time"] = (
                (api_stats["average_response_time"] * (api_stats["successful_requests"] - 1) + response_time)
                / api_stats["successful_requests"]
            )
            
            return {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "filename": file.filename,
                "processing_time": response_time
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        api_stats["failed_requests"] += 1
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=WORKERS,
        timeout_keep_alive=TIMEOUT
    ) 