# Slide Classification Training

A PyTorch-based image classification system optimized for M4 Mac mini with Metal acceleration. This project trains a MobileNetV4 model to classify different types of slide presentations and screen states.

## Two Training Approaches

This project provides two training and inference implementations:

### 1. Standard Version (PIL-based)
- **Files**: `train.py`, `inference.py`
- **Preprocessing**: PIL/Pillow with torchvision transforms
- **Use Case**: Standard PyTorch workflow, faster development
- **Model Output**: `slide_classifier_mobilenetv4.pth`

### 2. OpenCV Version (Cross-platform Compatible)
- **Files**: `train_opencv.py`, `inference_opencv.py`
- **Preprocessing**: OpenCV with `cv2.INTER_AREA` interpolation
- **Use Case**: Production deployment requiring identical results across Python, JavaScript, and C++
- **Model Output**: `slide_classifier_mobilenetv4_opencv.pth`
- **Key Benefit**: 100% consistent preprocessing across all platforms

**Recommendation**: Use the OpenCV version if you need to deploy the model in C++ or JavaScript environments, or if you require exact reproducibility across different platforms.

## Dataset Structure

The dataset contains 7 classes with significant imbalance:

```
dataset/
├── may_be_slide_powerpoint_edit_mode/
├── may_be_slide_powerpoint_side_screen/
├── not_slide_black_or_blue_screen/
├── not_slide_desktop/
├── not_slide_no_signal/
├── not_slide_others/
└── slide/
```

## Environment Setup

### Standard Training (PIL-based)

For standard training with PIL/Pillow preprocessing:

```bash
conda create --name trainning python=3.11
conda activate trainning
conda install pytorch torchvision torchaudio -c pytorch
pip install "numpy<2.0"
pip install transformers datasets timm pillow scikit-learn
```

### OpenCV Training (Cross-platform Compatible)

For training with OpenCV preprocessing (ensures identical results across Python, JavaScript, and C++):

```bash
conda create --name trainning python=3.11
conda activate trainning
conda install pytorch torchvision torchaudio -c pytorch
pip install "numpy<2.0"
pip install transformers datasets timm scikit-learn
pip install "opencv-python<4.10.0"
```

**Note**: The OpenCV version uses `cv2.INTER_AREA` for resizing, which matches C++ implementations exactly. Use this version if you need cross-platform consistency. We use `opencv-python<4.10.0` to ensure compatibility with `numpy<2.0` required by PyTorch.

## Model Architecture

- **Base Model**: MobileNetV4 (timm/mobilenetv4_conv_medium.e500_r256_in1k)
- **Input Size**: 256×256 pixels
- **Pretrained**: Yes (ImageNet)
- **Acceleration**: Metal Performance Shaders (MPS) for M4 Mac

## Key Features

### 1. Metal Acceleration & Performance
- **MPS Backend**: Automatically detects and uses Metal Performance Shaders for M4 Mac
- **Optimized Batch Size**: Default 64 (up from 32) for better GPU utilization
- **Multi-threaded Data Loading**: Default 4 workers for faster data loading
- **Mixed Precision Training**: Automatic mixed precision for better performance
- **Fast Mode**: `--fast_mode` enables batch_size=96, num_workers=6 for maximum speed

### 2. Dynamic Class Weighting
- Automatically calculates class weights based on dataset distribution
- Addresses severe class imbalance (4 images vs 2530 images)
- Uses inverse frequency weighting

### 3. Minimal Data Augmentation
- Only color, brightness, and contrast jitter (as requested)
- No rotation or cropping to preserve slide characteristics
- Maintains aspect ratio and content integrity

### 4. Early Stopping & Fine-tuning
- **Early Stopping**: Automatically stops training when validation loss stops improving
- **Fine-tuning**: Resume training from saved checkpoints with full state restoration
- **Smart Checkpointing**: Saves model, optimizer, scheduler, and training history

### 5. Comprehensive Monitoring
- Real-time training metrics
- Detailed classification reports
- Confusion matrix analysis
- Training history logging

## Usage

### Training

#### Standard Training (PIL-based)

**Basic Training:**
```bash
python train.py
```
*Note: If a previous model exists, the script will automatically detect it and ask if you want to resume training.*

**Resume Training (Fine-tuning):**
```bash
python train.py --resume models/slide_classifier_mobilenetv4.pth
```

**Force Fresh Training:**
```bash
python train.py --force_fresh
```

**Fast Mode (Recommended for M4 Mac):**
```bash
python train.py --fast_mode
```

**Custom Configuration:**
```bash
python train.py --batch_size 96 --num_workers 6 --learning_rate 0.0005 --early_stopping_patience 15
```

#### OpenCV Training (Cross-platform Compatible)

Use `train_opencv.py` for training with OpenCV preprocessing. This ensures 100% identical preprocessing across Python, JavaScript, and C++ implementations.

**Basic Training:**
```bash
python train_opencv.py
```

**Fast Mode (Recommended for M4 Mac):**
```bash
python train_opencv.py --fast_mode
```

**Custom Configuration:**
```bash
python train_opencv.py --batch_size 96 --num_workers 6 --learning_rate 0.0005 --early_stopping_patience 15
```

**Key Differences:**
- Uses `cv2.INTER_AREA` interpolation (matches C++ exactly)
- Custom OpenCV-based color augmentation
- Saves to `slide_classifier_mobilenetv4_opencv.pth` by default
- Includes preprocessing metadata in checkpoint

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | None | Path to checkpoint to resume training from |
| `--data_dir` | 'dataset' | Path to dataset directory |
| `--batch_size` | 64 | Batch size for training (increased for better GPU utilization) |
| `--num_epochs` | 50 | Number of epochs to train |
| `--learning_rate` | 0.001 | Learning rate |
| `--weight_decay` | 1e-4 | Weight decay |
| `--early_stopping_patience` | 10 | Early stopping patience (epochs) |
| `--save_dir` | 'models' | Directory to save models |
| `--model_name` | 'slide_classifier_mobilenetv4.pth' | Model filename |
| `--force_fresh` | False | Force fresh training without prompting to resume |
| `--num_workers` | 4 | Number of data loading workers (0=single-threaded) |
| `--fast_mode` | False | Enable fast mode with optimized settings for M4 Mac |

### Configuration

The script uses these default settings:

```python
config = {
    'data_dir': 'dataset',
    'batch_size': 32,          # Optimized for 16GB RAM
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': 'models',
    'model_name': 'slide_classifier_mobilenetv4.pth'
}
```

### Output Files

After training, you'll find:

- `models/slide_classifier_mobilenetv4.pth` - Best model checkpoint
- `models/training_history.json` - Training metrics and configuration

## Model Performance

The script provides detailed metrics including:

- Training/validation loss and accuracy per epoch
- Classification report with precision, recall, F1-score
- Confusion matrix for detailed class analysis
- Best model selection based on validation accuracy

## Data Augmentation Strategy

Based on the business requirements:

```python
transforms.ColorJitter(
    brightness=0.1,    # Slight brightness variation
    contrast=0.1,      # Slight contrast variation
    saturation=0.1,    # Slight color saturation variation
    hue=0.05          # Very slight hue variation
)
```

**No rotation or cropping** to preserve slide content and layout.

## Class Weight Calculation

The script dynamically calculates weights using:

```
weight_i = total_samples / (num_classes × class_i_samples)
```

## Performance Optimizations

### Standard Mode (Default)
- **Device**: Metal Performance Shaders (MPS)
- **Batch Size**: 64 (optimized for 16GB RAM)
- **Workers**: 4 (multi-threaded data loading)
- **Pin Memory**: Enabled for faster GPU transfers
- **Mixed Precision**: Automatic for better performance

### Fast Mode (`--fast_mode`)
- **Batch Size**: 96 (maximum GPU utilization)
- **Workers**: 6 (aggressive multi-threading)
- **Expected Speedup**: 40-60% faster training
- **Memory Usage**: ~6-8GB (well within 16GB limit)

### Performance Comparison
| Mode | Batch Size | Workers | Expected Time/Epoch | Memory Usage |
|------|------------|---------|-------------------|--------------|
| Conservative | 32 | 0 | ~3 minutes | ~4GB |
| Standard | 64 | 4 | ~2 minutes | ~5-6GB |
| Fast | 96 | 6 | ~1.2-1.5 minutes | ~6-8GB |

## Training Process

1. **Data Loading**: Automatic dataset discovery and class mapping
2. **Train/Val Split**: 80/20 split with stratification
3. **Model Loading**: Pretrained MobileNetV4 with custom classifier
4. **Training Loop**: Up to 50 epochs with early stopping (default: 10 patience)
5. **Learning Rate**: Adaptive reduction on plateau
6. **Model Saving**: Best model based on validation accuracy

## Early Stopping & Fine-tuning

### Early Stopping
- **Purpose**: Prevents overfitting and saves training time
- **Default Patience**: 10 epochs (configurable with `--early_stopping_patience`)
- **Monitoring**: Validation loss with minimum improvement threshold (0.001)
- **Behavior**: Automatically restores best model weights when stopping

### Fine-tuning (Resume Training)
- **Automatic Detection**: Script automatically detects existing models and prompts to resume
- **Manual Resume**: Use `--resume path/to/model.pth` to specify a checkpoint
- **Force Fresh**: Use `--force_fresh` to skip resume prompt and start fresh
- **State Restoration**: Restores model, optimizer, scheduler, and training history
- **Benefits**:
  - Continue training with new data
  - Adjust hyperparameters mid-training
  - Recover from interrupted training sessions

### Model Naming Convention
- **Default Name**: `slide_classifier_mobilenetv4.pth`
- **Location**: `models/` directory (configurable with `--save_dir`)
- **Custom Name**: Use `--model_name your_model.pth` for custom naming

### Checkpoint Contents
Each saved model includes:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Scheduler state (`scheduler_state_dict`)
- Training metrics history
- Class mappings and configuration
- Best validation accuracy achieved

## Monitoring Training

The script outputs:
- Real-time batch progress
- Epoch summaries with loss/accuracy
- Detailed classification reports
- Best model updates

## Business Context Considerations

The model accounts for the specific use cases:

- **not_slide_no_signal**: Same image with possible distortion
- **not_slide_desktop**: Same wallpaper, varying icon layouts
- **not_slide_black_or_blue_screen**: Solid/near-solid color screens

The minimal augmentation strategy preserves these characteristics while providing slight robustness to recording variations.

## Troubleshooting

### Memory Issues
- **If out of memory**: Reduce `--batch_size` from 64 to 32 or 16
- **If still issues**: Use `--num_workers 0` to disable multiprocessing
- **Conservative mode**: `python train.py --batch_size 32 --num_workers 0`

### File Handle Issues
- **If "too many open files"**: Reduce `--num_workers` or set to 0
- **macOS specific**: The script handles this automatically in most cases

### MPS Issues
- Script will automatically fall back to CPU if MPS unavailable
- Ensure macOS and PyTorch are up to date

### Dataset Issues
- Ensure all images are readable (PNG/JPG format)
- Check for corrupted files if training fails

## Next Steps

After training:

1. Evaluate model performance on validation set
2. Test on new screen recordings
3. Consider data collection for underrepresented classes
4. Fine-tune hyperparameters based on results

## Model Inference

### Using the Inference Script (Recommended)

#### Standard Inference (PIL-based)

The project includes a comprehensive inference script (`inference.py`) for easy model usage:

**Basic Usage:**
```bash
# Process all images in test directory (default)
python inference.py

# Process a single image
python inference.py --input path/to/image.png

# Process a specific directory
python inference.py --input path/to/directory

# Save results to JSON file
python inference.py --output results.json

# Show detailed probabilities for all classes
python inference.py --show_probabilities

# Set confidence threshold
python inference.py --confidence_threshold 0.7
```

#### OpenCV Inference (Cross-platform Compatible)

Use `inference_opencv.py` for inference with OpenCV preprocessing. This ensures consistent results with models trained using `train_opencv.py`.

**Basic Usage:**
```bash
# Process all images in test directory (default)
python inference_opencv.py

# Process a single image
python inference_opencv.py --input path/to/image.png

# Process a specific directory
python inference_opencv.py --input path/to/directory

# Use OpenCV-trained model
python inference_opencv.py --model models/slide_classifier_mobilenetv4_opencv.pth

# Save results to JSON file
python inference_opencv.py --output results.json

# Show detailed probabilities for all classes
python inference_opencv.py --show_probabilities
```

**Important**: Always use `inference_opencv.py` with models trained using `train_opencv.py` to ensure preprocessing consistency.

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `models/slide_classifier_mobilenetv4.pth` | Path to trained model checkpoint |
| `--input` | `test` | Path to input image or directory |
| `--output` | None | Path to save results JSON file |
| `--confidence_threshold` | 0.5 | Minimum confidence threshold for predictions |
| `--show_probabilities` | False | Show probabilities for all classes |
| `--max_display` | 20 | Maximum number of detailed results to display (0 for all) |
| `--quiet` | False | Only show summary, not detailed results |

**Example Output:**
```
Using device: mps
Loading model from: models/slide_classifier_mobilenetv4.pth
Found 7 classes: ['may_be_slide_powerpoint_edit_mode', 'may_be_slide_powerpoint_side_screen', 'not_slide_black_or_blue_screen', 'not_slide_desktop', 'not_slide_no_signal', 'not_slide_others', 'slide']
Model trained for 25 epochs, best validation accuracy: 94.32%
Found 37 images in test
Processing image 1/37: Slide_1760272553436.png

============================================================
PREDICTION SUMMARY
============================================================
Total images processed: 37
Successful predictions: 37
Errors: 0
Above confidence threshold: 35/37
Average confidence: 0.892

Class Distribution:
----------------------------------------
slide                               32 ( 86.5%)
not_slide_others                     3 (  8.1%)
not_slide_desktop                    2 (  5.4%)
```

### Manual Inference (Advanced)

For custom integration, you can use the model directly:

```python
import torch
import timm
from PIL import Image
from torchvision import transforms

# Load model
checkpoint = torch.load('models/slide_classifier_mobilenetv4.pth')
model = timm.create_model("hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k",
                         pretrained=False, num_classes=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('path/to/image.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(1).item()

class_names = checkpoint['class_names']
print(f"Predicted: {class_names[predicted_class]}")
```

### Inference Features

The inference script provides:

1. **Batch Processing**: Process entire directories of images efficiently
2. **Confidence Scoring**: Get confidence scores for predictions with customizable thresholds
3. **Class Probabilities**: View probabilities for all classes, not just the top prediction
4. **Error Handling**: Graceful handling of corrupted or unreadable images
5. **Performance Metrics**: Processing time and throughput statistics
6. **JSON Export**: Save results for further analysis or integration
7. **Metal Acceleration**: Automatic MPS (Metal Performance Shaders) support for M4 Mac

## ONNX Model Conversion

The project includes a conversion script to export the trained PyTorch model to ONNX format for deployment and inference in production environments.

### Converting to ONNX

**Basic Conversion:**
```bash
python convert_to_onnx.py
```

**Custom Configuration:**
```bash
python convert_to_onnx.py --model_path models/slide_classifier_mobilenetv4.pth \
                          --output_dir models \
                          --output_name slide_classifier_mobilenetv4.onnx \
                          --verify
```

**With Quantization (Recommended for Production):**
```bash
# INT8 quantization (smaller size, faster inference)
python convert_to_onnx.py --quantization int8

# UINT8 quantization
python convert_to_onnx.py --quantization uint8

# FP16 quantization (half precision)
python convert_to_onnx.py --quantization fp16
```

**Skip Verification (faster):**
```bash
python convert_to_onnx.py --no_verify
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `models/slide_classifier_mobilenetv4.pth` | Path to the trained PyTorch model |
| `--output_dir` | `models` | Output directory for ONNX model |
| `--output_name` | `slide_classifier_mobilenetv4.onnx` | Output ONNX model filename |
| `--opset_version` | 11 | ONNX opset version |
| `--verify` | True | Verify ONNX model against PyTorch model |
| `--no_verify` | False | Skip ONNX model verification |
| `--quantization` | `none` | Quantization type: `none`, `int8`, `uint8`, `fp16` |

### Output Files

After conversion, you'll find:

- `models/slide_classifier_mobilenetv4.onnx` - ONNX model file (unquantized)
- `models/slide_classifier_mobilenetv4_int8.onnx` - INT8 quantized model (if using `--quantization int8`)
- `models/slide_classifier_mobilenetv4_uint8.onnx` - UINT8 quantized model (if using `--quantization uint8`)
- `models/slide_classifier_mobilenetv4_fp16.onnx` - FP16 quantized model (if using `--quantization fp16`)
- `models/model_info.json` - Model metadata and configuration

### Model Information

The conversion script generates a `model_info.json` file containing:

```json
{
  "class_names": ["may_be_slide_powerpoint_edit_mode", "may_be_slide_powerpoint_side_screen", ...],
  "num_classes": 7,
  "input_shape": [1, 3, 256, 256],
  "input_size": [256, 256],
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "model_architecture": "mobilenetv4_conv_medium.e500_r256_in1k",
  "quantization": "int8",
  "training_info": {
    "best_val_acc": 94.32,
    "epoch": 25,
    "config": {...}
  }
}
```

### Using ONNX Model

**Python Example:**
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import json

# Load model info
with open('models/model_info.json', 'r') as f:
    model_info = json.load(f)

# Create ONNX Runtime session
session = ort.InferenceSession('models/slide_classifier_mobilenetv4.onnx')

# Prepare image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model_info['normalization']['mean'],
        std=model_info['normalization']['std']
    )
])

# Load and preprocess image
image = Image.open('path/to/image.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0).numpy()

# Run inference
outputs = session.run(None, {'input': input_tensor})
predictions = outputs[0]

# Get predicted class
predicted_class_idx = np.argmax(predictions, axis=1)[0]
predicted_class = model_info['class_names'][predicted_class_idx]
confidence = np.max(predictions)

print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
```

### ONNX Model Features

1. **Cross-Platform Compatibility**: Deploy on various platforms and frameworks
2. **Optimized Inference**: Better performance for production deployment
3. **Smaller File Size**: Typically smaller than PyTorch checkpoints
4. **Framework Agnostic**: Use with ONNX Runtime, TensorRT, OpenVINO, etc.
5. **Model Verification**: Automatic verification ensures output consistency
6. **Metadata Preservation**: All training information and class mappings preserved
7. **Quantization Support**: Multiple quantization options for optimized deployment

### Quantization Options

The conversion script supports several quantization methods to optimize model size and inference speed:

#### INT8 Quantization (Recommended)
- **File Size**: ~75% smaller than FP32
- **Speed**: 2-4x faster inference
- **Accuracy**: Minimal accuracy loss (<1%)
- **Use Case**: Production deployment, edge devices

#### UINT8 Quantization
- **File Size**: ~75% smaller than FP32
- **Speed**: 2-4x faster inference
- **Accuracy**: Similar to INT8
- **Use Case**: Specific hardware requirements

#### FP16 Quantization (Half Precision)
- **File Size**: ~50% smaller than FP32
- **Speed**: 1.5-2x faster inference
- **Accuracy**: Negligible accuracy loss
- **Use Case**: GPU deployment, balanced size/accuracy

#### No Quantization (Default)
- **File Size**: Full precision (largest)
- **Speed**: Baseline inference speed
- **Accuracy**: Maximum accuracy
- **Use Case**: Development, maximum precision requirements

**Installation Requirements for Quantization:**
```bash
pip install onnxruntime onnxconverter-common
```

**Important Notes for Quantized Models:**
- Quantized models may not be verifiable on all systems due to ONNX Runtime limitations
- The conversion process will succeed, but verification might fail with "ConvInteger not implemented"
- This is normal behavior - the quantized model is still valid and usable
- For deployment, ensure your target environment supports quantized ONNX operators
- Consider using ONNX Runtime with CPU execution provider (MLAS) for quantized model inference

### Deployment Options

The ONNX model can be deployed using:

- **ONNX Runtime**: Cross-platform inference (recommended)
- **TensorRT**: NVIDIA GPU acceleration
- **OpenVINO**: Intel hardware optimization
- **Core ML**: Apple device deployment
- **Web Deployment**: ONNX.js for browser inference
- **Mobile**: ONNX Runtime Mobile for iOS/Android

### Performance Considerations

- **Input Shape**: Fixed at (1, 3, 256, 256) but supports dynamic batch sizes
- **Preprocessing**: Ensure proper ImageNet normalization
- **Memory Usage**: ~25-30MB model size
- **Inference Speed**: Typically 2-5x faster than PyTorch for single image inference