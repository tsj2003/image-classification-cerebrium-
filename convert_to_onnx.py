import argparse
import torch
from src.model.pytorch_model import PyTorchModel

def convert_to_onnx(model_path: str, output_path: str, include_preprocessing: bool = True):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to PyTorch model weights
        output_path (str): Path to save ONNX model
        include_preprocessing (bool): Whether to include preprocessing in ONNX model
    """
    # Load PyTorch model
    model = PyTorchModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, default='pytorch_model_weights.pth',
                      help='Path to PyTorch model weights')
    parser.add_argument('--output_path', type=str, default='model.onnx',
                      help='Path to save ONNX model')
    parser.add_argument('--include_preprocessing', action='store_true',
                      help='Include preprocessing in ONNX model')
    
    args = parser.parse_args()
    convert_to_onnx(args.model_path, args.output_path, args.include_preprocessing)

if __name__ == '__main__':
    main() 