import os
import sys
import torch
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pytorch_model import Classifier

def convert_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 12
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to the PyTorch model weights
        output_path (str): Path where the ONNX model will be saved
        input_shape (tuple): Input shape for the model (batch_size, channels, height, width)
        opset_version (int): ONNX opset version to use
    """
    # Initialize model
    model = Classifier()
    
    # Load weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export the model
    torch.onnx.export(
        model,                     # model being run
        dummy_input,              # model input (or a tuple for multiple inputs)
        output_path,              # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model has been converted to ONNX and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the PyTorch model weights')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path where the ONNX model will be saved')
    parser.add_argument('--opset_version', type=int, default=12,
                      help='ONNX opset version to use')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    convert_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        opset_version=args.opset_version
    )

if __name__ == '__main__':
    main() 