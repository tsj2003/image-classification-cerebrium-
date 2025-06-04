import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from src.model.pytorch_model import PyTorchModel

class PreprocessingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resize = transforms.Resize((224, 224))
        self.crop = transforms.CenterCrop((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # Input is expected to be a PIL Image
        x = self.resize(x)
        x = self.crop(x)
        x = self.to_tensor(x)
        x = self.normalize(x)
        x = x.unsqueeze(0)  # Add batch dimension
        return self.model(x)

def convert_to_onnx(model_path: str, output_path: str, include_preprocessing: bool = True):
    """
    Convert PyTorch model to ONNX format with optional preprocessing steps.
    
    Args:
        model_path (str): Path to PyTorch model weights
        output_path (str): Path to save ONNX model
        include_preprocessing (bool): Whether to include preprocessing in ONNX model
    """
    # Load PyTorch model
    model = PyTorchModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    if include_preprocessing:
        # Wrap model with preprocessing
        model = PreprocessingModel(model)
        # Create dummy input (PIL Image)
        dummy_input = Image.new('RGB', (224, 224), color='red')
    else:
        # Create dummy input tensor
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
    if include_preprocessing:
        print("Preprocessing steps are included in the ONNX model")

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