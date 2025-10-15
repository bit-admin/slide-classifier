#!/usr/bin/env python3
"""
Convert trained MobileNetV4 slide classifier to ONNX format
"""

import os
import torch
import torch.nn as nn
import timm
import argparse
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image

def create_model(num_classes):
    """Create MobileNetV4 model - same as in train.py"""
    print("Loading MobileNetV4 model...")

    # Load pretrained MobileNetV4
    model = timm.create_model(
        "hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k",
        pretrained=True,
        num_classes=num_classes
    )

    return model

def load_trained_model(checkpoint_path, device='cpu'):
    """Load the trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)

    if num_classes == 0:
        raise ValueError("No class names found in checkpoint. Cannot determine number of classes.")

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Create model
    model = create_model(num_classes)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_names, checkpoint

def convert_to_onnx(model, output_path, input_shape=(1, 3, 256, 256), opset_version=11):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting model to ONNX format...")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {output_path}")

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model successfully exported to: {output_path}")

def verify_onnx_model(onnx_path, pytorch_model, test_input=None):
    """Verify that ONNX model produces same outputs as PyTorch model"""
    print("Verifying ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Create test input if not provided
    if test_input is None:
        test_input = torch.randn(1, 3, 256, 256)

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)

    # Get ONNX output
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    pytorch_output_np = pytorch_output.numpy()
    max_diff = np.max(np.abs(pytorch_output_np - onnx_output))

    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")

    if max_diff < 1e-5:
        print("✓ ONNX model verification successful - outputs match!")
        return True
    else:
        print("⚠ Warning: Outputs differ significantly")
        return False

def create_test_image_tensor():
    """Create a test image tensor with proper preprocessing"""
    # Create a dummy RGB image (256x256)
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # Apply same transforms as in training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform and add batch dimension
    tensor = transform(test_image).unsqueeze(0)
    return tensor

def save_model_info(output_dir, class_names, checkpoint_info):
    """Save model information for inference"""
    import json

    model_info = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'input_shape': [1, 3, 256, 256],
        'input_size': [256, 256],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'model_architecture': 'mobilenetv4_conv_medium.e500_r256_in1k',
        'training_info': {
            'best_val_acc': checkpoint_info.get('best_val_acc', 'unknown'),
            'epoch': checkpoint_info.get('epoch', 'unknown'),
            'config': checkpoint_info.get('config', {})
        }
    }

    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model information saved to: {info_path}")
    return info_path

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch slide classifier to ONNX')
    parser.add_argument('--model_path', type=str,
                        default='models/slide_classifier_mobilenetv4.pth',
                        help='Path to the trained PyTorch model')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for ONNX model')
    parser.add_argument('--output_name', type=str,
                        default='slide_classifier_mobilenetv4.onnx',
                        help='Output ONNX model filename')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX model against PyTorch model')
    parser.add_argument('--no_verify', action='store_true',
                        help='Skip ONNX model verification')

    args = parser.parse_args()

    # Override verify flag if no_verify is set
    if args.no_verify:
        args.verify = False

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Make sure you have trained the model first using train.py")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load trained model
        model, class_names, checkpoint_info = load_trained_model(args.model_path)

        # Set output path
        output_path = os.path.join(args.output_dir, args.output_name)

        # Convert to ONNX
        convert_to_onnx(
            model,
            output_path,
            input_shape=(1, 3, 256, 256),
            opset_version=args.opset_version
        )

        # Verify model if requested
        if args.verify:
            test_input = create_test_image_tensor()
            verify_onnx_model(output_path, model, test_input)

        # Save model information
        info_path = save_model_info(args.output_dir, class_names, checkpoint_info)

        print(f"\n✓ Conversion completed successfully!")
        print(f"ONNX model: {output_path}")
        print(f"Model info: {info_path}")
        print(f"Classes ({len(class_names)}): {class_names}")

        # Print usage example
        print(f"\nUsage example:")
        print(f"import onnxruntime as ort")
        print(f"session = ort.InferenceSession('{output_path}')")
        print(f"# Preprocess your image to shape (1, 3, 256, 256) with ImageNet normalization")
        print(f"# output = session.run(None, {{'input': preprocessed_image}})[0]")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()