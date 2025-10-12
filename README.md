# Slide Classification Training

A PyTorch-based image classification system optimized for M4 Mac mini with Metal acceleration. This project trains a MobileNetV4 model to classify different types of slide presentations and screen states.

## Dataset Structure

The dataset contains 7 classes with significant imbalance:

```
dataset/
├── may_be_slide_powerpoint_edit_mode/     (36 images)
├── may_be_slide_powerpoint_side_screen/   (15 images)
├── not_slide_black:blue_screen/           (4 images)
├── not_slide_desktop/                     (62 images)
├── not_slide_no_signal/                   (26 images)
├── not_slide_others/                      (479 images)
└── slide/                                 (2530 images)
```

**Total: 3,152 images across 7 classes**

## Environment Setup

You've already set up the environment. For reference:

```bash
conda create --name trainning python=3.11
conda activate trainning
conda install pytorch torchvision torchaudio -c pytorch
pip install "numpy<2.0"
pip install transformers datasets timm pillow scikit-learn
```

## Model Architecture

- **Base Model**: MobileNetV4 (timm/mobilenetv4_conv_medium.e500_r256_in1k)
- **Input Size**: 256×256 pixels
- **Pretrained**: Yes (ImageNet)
- **Acceleration**: Metal Performance Shaders (MPS) for M4 Mac

## Key Features

### 1. Metal Acceleration
- Automatically detects and uses MPS backend for M4 Mac optimization
- Optimized batch size (32) for 16GB RAM
- Single-threaded data loading to avoid file handle issues

### 2. Dynamic Class Weighting
- Automatically calculates class weights based on dataset distribution
- Addresses severe class imbalance (4 images vs 2530 images)
- Uses inverse frequency weighting

### 3. Minimal Data Augmentation
- Only color, brightness, and contrast jitter (as requested)
- No rotation or cropping to preserve slide characteristics
- Maintains aspect ratio and content integrity

### 4. Comprehensive Monitoring
- Real-time training metrics
- Detailed classification reports
- Confusion matrix analysis
- Training history logging

## Usage

### Training

```bash
python train.py
```

### Configuration

The script uses these default settings (modify in `train.py` if needed):

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

Based on your business requirements:

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

Current weights for your dataset:
- `may_be_slide_powerpoint_edit_mode`: 12.51×
- `may_be_slide_powerpoint_side_screen`: 30.02×
- `not_slide_black:blue_screen`: 112.57× (highest due to only 4 samples)
- `not_slide_desktop`: 7.26×
- `not_slide_no_signal`: 17.32×
- `not_slide_others`: 0.94×
- `slide`: 0.18× (lowest due to 2530 samples)

## Hardware Optimization

Optimized for M4 Mac mini (10-core, 16GB):

- **Device**: Metal Performance Shaders (MPS)
- **Batch Size**: 32 (memory efficient)
- **Workers**: 0 (single-threaded to avoid file handle issues)
- **Pin Memory**: Disabled (not needed with single-threaded loading)

## Training Process

1. **Data Loading**: Automatic dataset discovery and class mapping
2. **Train/Val Split**: 80/20 split with stratification
3. **Model Loading**: Pretrained MobileNetV4 with custom classifier
4. **Training Loop**: 50 epochs with early stopping potential
5. **Learning Rate**: Adaptive reduction on plateau
6. **Model Saving**: Best model based on validation accuracy

## Expected Training Time

On M4 Mac mini:
- ~2-3 minutes per epoch
- Total training time: ~1.5-2.5 hours for 50 epochs

## Monitoring Training

The script outputs:
- Real-time batch progress
- Epoch summaries with loss/accuracy
- Detailed classification reports
- Best model updates

## Business Context Considerations

The model accounts for your specific use cases:

- **not_slide_no_signal**: Same image with possible distortion
- **not_slide_desktop**: Same wallpaper, varying icon layouts
- **not_slide_black:blue_screen**: Solid/near-solid color screens

The minimal augmentation strategy preserves these characteristics while providing slight robustness to recording variations.

## Troubleshooting

### Memory Issues
- Reduce `batch_size` from 32 to 16 or 8
- The script already uses single-threaded loading (`num_workers=0`) to avoid file handle issues

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

To use the trained model:

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