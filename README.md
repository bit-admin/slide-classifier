# Slide Classification Training

A PyTorch-based image classification system optimized for M4 Mac mini with Metal acceleration. This project trains a MobileNetV4 model to classify different types of slide presentations and screen states.

## Dataset Structure

The dataset contains 7 classes with significant imbalance:

```
dataset/
├── may_be_slide_powerpoint_edit_mode/     (36 images)
├── may_be_slide_powerpoint_side_screen/   (15 images)
├── not_slide_black_or_blue_screen/           (4 images)
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
- `not_slide_black_or_blue_screen`: 112.57× (highest due to only 4 samples)
- `not_slide_desktop`: 7.26×
- `not_slide_no_signal`: 17.32×
- `not_slide_others`: 0.94×
- `slide`: 0.18× (lowest due to 2530 samples)

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

## Expected Training Time

On M4 Mac mini with different modes:

### Standard Mode (Default)
- **Per Epoch**: ~2 minutes
- **With Early Stopping**: 15-25 epochs (~30-50 minutes)
- **Full 50 epochs**: ~1.5 hours

### Fast Mode (`--fast_mode`)
- **Per Epoch**: ~1.2-1.5 minutes
- **With Early Stopping**: 15-25 epochs (~18-38 minutes)
- **Full 50 epochs**: ~1 hour

### Conservative Mode (if needed)
- **Per Epoch**: ~3 minutes
- **With Early Stopping**: 15-25 epochs (~45-75 minutes)
- **Full 50 epochs**: ~2.5 hours

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