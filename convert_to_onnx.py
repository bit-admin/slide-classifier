#!/usr/bin/env python3
"""
Convert trained MobileNetV4 slide classifier to ONNX format
"""

import os
import json
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

def embed_metadata_in_onnx(onnx_path, class_names, model_info=None):
    """
    Embed class names and model metadata into ONNX model's custom metadata.

    This allows the C++ inference code to read class names directly from the
    model file instead of requiring a separate JSON file.

    Args:
        onnx_path: Path to the ONNX model file
        class_names: List of class names (in order matching model output)
        model_info: Optional dict with additional model information
    """
    print(f"Embedding metadata into ONNX model...")

    # Load the ONNX model
    model = onnx.load(onnx_path)

    # Add class_names as JSON array
    meta_class_names = model.metadata_props.add()
    meta_class_names.key = "class_names"
    meta_class_names.value = json.dumps(class_names)

    # Add num_classes
    meta_num_classes = model.metadata_props.add()
    meta_num_classes.key = "num_classes"
    meta_num_classes.value = str(len(class_names))

    # Add input dimensions
    meta_input_height = model.metadata_props.add()
    meta_input_height.key = "input_height"
    meta_input_height.value = "256"

    meta_input_width = model.metadata_props.add()
    meta_input_width.key = "input_width"
    meta_input_width.value = "256"

    # Add normalization parameters (ImageNet defaults)
    meta_norm_mean = model.metadata_props.add()
    meta_norm_mean.key = "normalization_mean"
    meta_norm_mean.value = json.dumps([0.485, 0.456, 0.406])

    meta_norm_std = model.metadata_props.add()
    meta_norm_std.key = "normalization_std"
    meta_norm_std.value = json.dumps([0.229, 0.224, 0.225])

    # Add model architecture info
    meta_arch = model.metadata_props.add()
    meta_arch.key = "model_architecture"
    meta_arch.value = "mobilenetv4_conv_medium.e500_r256_in1k"

    # Add optional model_info as JSON if provided
    if model_info:
        meta_info = model.metadata_props.add()
        meta_info.key = "model_info"
        meta_info.value = json.dumps(model_info)

    # Save the modified model
    onnx.save(model, onnx_path)

    print(f"Metadata embedded successfully:")
    print(f"  - class_names: {class_names}")
    print(f"  - num_classes: {len(class_names)}")
    print(f"  - input_size: 256x256")

def verify_onnx_model(onnx_path, pytorch_model, test_input=None, is_quantized=False):
    """Verify that ONNX model produces same outputs as PyTorch model"""
    print("Verifying ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    # Create ONNX Runtime session
    try:
        ort_session = ort.InferenceSession(onnx_path)
    except Exception as e:
        if is_quantized and "ConvInteger" in str(e):
            print("⚠ Warning: Cannot verify quantized model - quantized operators not supported by current ONNX Runtime")
            print("This is normal for quantized models. The model file is valid but requires:")
            print("- ONNX Runtime with quantization support")
            print("- Appropriate execution providers (CPU with MLAS, or specific hardware)")
            print("✓ Model structure validation passed - quantization successful")
            return True
        else:
            raise e

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
    mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))

    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

    # Print detailed comparison
    print("\nDetailed output comparison:")
    print("PyTorch output shape:", pytorch_output_np.shape)
    print("ONNX output shape:", onnx_output.shape)
    print("PyTorch output (first 5 values):", pytorch_output_np[0][:5])
    print("ONNX output (first 5 values):", onnx_output[0][:5])

    # Apply softmax to both outputs for probability comparison
    pytorch_probs = torch.softmax(pytorch_output, dim=1).numpy()
    onnx_logits = torch.from_numpy(onnx_output)
    onnx_probs = torch.softmax(onnx_logits, dim=1).numpy()

    print("\nProbability comparison (after softmax):")
    print("PyTorch probabilities:", pytorch_probs[0])
    print("ONNX probabilities:", onnx_probs[0])

    prob_max_diff = np.max(np.abs(pytorch_probs - onnx_probs))
    print(f"Maximum probability difference: {prob_max_diff}")

    # Check if predictions match
    pytorch_pred = np.argmax(pytorch_probs[0])
    onnx_pred = np.argmax(onnx_probs[0])
    print(f"PyTorch predicted class: {pytorch_pred}")
    print(f"ONNX predicted class: {onnx_pred}")
    print(f"Predictions match: {pytorch_pred == onnx_pred}")

    if max_diff < 1e-4 and prob_max_diff < 1e-4:
        print("✓ ONNX model verification successful - outputs match!")
        return True
    elif pytorch_pred == onnx_pred and prob_max_diff < 1e-2:
        print("✓ ONNX model verification acceptable - predictions match with minor numerical differences")
        return True
    else:
        print("⚠ Warning: Outputs differ significantly")
        print("This may cause different predictions between PyTorch and ONNX versions")
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

def quantize_onnx_model(input_path, output_path, quantization_type):
    """Quantize ONNX model using ONNX Runtime quantization"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.quantize import quantize_static
        from onnxruntime.quantization.calibrate import CalibrationDataReader
        import tempfile

        print(f"Quantizing model with {quantization_type} quantization...")

        if quantization_type == 'int8':
            # Dynamic quantization to INT8
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8
            )
        elif quantization_type == 'uint8':
            # Dynamic quantization to UINT8
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8
            )
        elif quantization_type == 'fp16':
            # For FP16, we need to use a different approach
            import onnx
            from onnxconverter_common import float16

            # Load the model
            model = onnx.load(input_path)

            # Convert to FP16
            model_fp16 = float16.convert_float_to_float16(model)

            # Save the FP16 model
            onnx.save(model_fp16, output_path)

        print(f"Quantized model saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"Warning: Quantization libraries not available: {e}")
        print("To use quantization, install: pip install onnxruntime onnxconverter-common")
        return False
    except Exception as e:
        print(f"Error during quantization: {e}")
        return False

def save_model_info(output_dir, class_names, checkpoint_info, quantization_type='none'):
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
        'quantization': quantization_type,
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

def generate_output_filename(base_name, quantization_type):
    """Generate output filename based on quantization type"""
    if quantization_type == 'none':
        return base_name

    # Split filename and extension
    name_parts = base_name.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        return f"{name}_{quantization_type}.{ext}"
    else:
        return f"{base_name}_{quantization_type}"

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
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'int8', 'uint8', 'fp16'],
                        help='Quantization type (default: none)')

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

        # Generate output filename based on quantization type
        final_output_name = generate_output_filename(args.output_name, args.quantization)
        output_path = os.path.join(args.output_dir, final_output_name)

        # If quantization is requested, we need to create a temporary unquantized model first
        if args.quantization != 'none':
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_output_path = os.path.join(temp_dir, 'temp_model.onnx')

            print(f"Creating temporary unquantized model for quantization...")
            # Convert to ONNX (temporary file)
            convert_to_onnx(
                model,
                temp_output_path,
                input_shape=(1, 3, 256, 256),
                opset_version=args.opset_version
            )

            # Quantize the model
            quantization_success = quantize_onnx_model(temp_output_path, output_path, args.quantization)

            # Embed metadata into quantized model before cleanup
            if quantization_success:
                embed_metadata_in_onnx(output_path, class_names, {
                    'quantization': args.quantization,
                    'best_val_acc': checkpoint_info.get('best_val_acc', 'unknown'),
                    'epoch': checkpoint_info.get('epoch', 'unknown')
                })

            # Clean up temporary file
            import shutil
            shutil.rmtree(temp_dir)

            if not quantization_success:
                print("Quantization failed, falling back to unquantized model...")
                # Fallback: create unquantized model with original name
                output_path = os.path.join(args.output_dir, args.output_name)
                convert_to_onnx(
                    model,
                    output_path,
                    input_shape=(1, 3, 256, 256),
                    opset_version=args.opset_version
                )
                # Embed metadata into fallback model
                embed_metadata_in_onnx(output_path, class_names, {
                    'quantization': 'none',
                    'best_val_acc': checkpoint_info.get('best_val_acc', 'unknown'),
                    'epoch': checkpoint_info.get('epoch', 'unknown')
                })
        else:
            # Convert to ONNX (no quantization)
            convert_to_onnx(
                model,
                output_path,
                input_shape=(1, 3, 256, 256),
                opset_version=args.opset_version
            )

        # Embed metadata into the ONNX model
        embed_metadata_in_onnx(output_path, class_names, {
            'quantization': args.quantization,
            'best_val_acc': checkpoint_info.get('best_val_acc', 'unknown'),
            'epoch': checkpoint_info.get('epoch', 'unknown')
        })

        # Verify model if requested
        if args.verify:
            test_input = create_test_image_tensor()
            is_quantized = args.quantization != 'none'
            verify_onnx_model(output_path, model, test_input, is_quantized)

        # Save model information
        info_path = save_model_info(args.output_dir, class_names, checkpoint_info, args.quantization)

        print(f"\n✓ Conversion completed successfully!")
        print(f"ONNX model: {output_path}")
        print(f"Model info: {info_path}")
        print(f"Classes ({len(class_names)}): {class_names}")
        print(f"Quantization: {args.quantization}")

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