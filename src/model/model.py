import os
from typing import List, Tuple, Union
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as transforms
import scipy.special

class ImagePreprocessor:
    """Handles image preprocessing for model input."""
    
    def __init__(self, target_size: tuple = (224, 224)):
        """Initialize preprocessor with target image size."""
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess an image for model input.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array with shape (1, 3, H, W)
        """
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize with mean and std
        image_array = (image_array - self.mean) / self.std
        
        # Transpose to (C, H, W) and add batch dimension
        image_array = image_array.transpose(2, 0, 1)[None, ...]
        
        return image_array.astype(np.float32)

class ONNXModel:
    """Handles ONNX model loading and inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize the ONNX model.
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, image_path: str) -> tuple:
        """Run inference on a single image and return class ID and confidence."""
        input_data = self.preprocessor.preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(
            [self.session.get_outputs()[0].name],
            {input_name: input_data}
        )
        logits = outputs[0][0]
        probs = scipy.special.softmax(logits)
        class_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return class_id, confidence

    def predict_batch(self, image_paths: list) -> list:
        """Run inference on a batch of images."""
        return [self.predict(img_path) for img_path in image_paths] 