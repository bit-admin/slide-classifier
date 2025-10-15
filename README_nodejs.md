# Node.js Slide Classification Inference

This document provides instructions for using the Node.js inference script with the ONNX model.

## Prerequisites

- Node.js 16.0.0 or higher
- npm (comes with Node.js)

## Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

   This will install:
   - `onnxruntime-node`: ONNX Runtime for Node.js
   - `sharp`: High-performance image processing
   - `commander`: Command-line argument parsing

## Usage

### Basic Usage

```bash
# Process all images in test directory (default)
node inference.js

# Process a single image
node inference.js --input path/to/image.png

# Process a specific directory
node inference.js --input path/to/directory

# Save results to JSON file
node inference.js --output results.json

# Show detailed probabilities for all classes
node inference.js --show-probabilities

# Set confidence threshold
node inference.js --confidence-threshold 0.7
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | `models/slide_classifier_mobilenetv4.onnx` | Path to ONNX model file |
| `--model-info <path>` | `models/model_info.json` | Path to model info JSON file |
| `--input <path>` | `test` | Path to input image or directory |
| `--output <path>` | None | Path to save results JSON file |
| `--confidence-threshold <number>` | 0.5 | Minimum confidence threshold for predictions |
| `--show-probabilities` | false | Show probabilities for all classes |
| `--max-display <number>` | 20 | Maximum number of detailed results to display (0 for all) |
| `--quiet` | false | Only show summary, not detailed results |

### Examples

**Process single image with detailed output:**
```bash
node inference.js --input test/slide_example.png --show-probabilities
```

**Batch process directory and save results:**
```bash
node inference.js --input test --output batch_results.json --confidence-threshold 0.8
```

**Quiet mode for automation:**
```bash
node inference.js --input test --quiet --output results.json
```

## Output Format

### Console Output

The script provides detailed console output including:

```
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

### JSON Output

When using `--output`, results are saved in JSON format:

```json
[
  {
    "image_path": "test/slide_example.png",
    "predicted_class": "slide",
    "confidence": 0.956,
    "above_threshold": true,
    "class_probabilities": {
      "slide": 0.956,
      "not_slide_others": 0.032,
      "not_slide_desktop": 0.008,
      "may_be_slide_powerpoint_edit_mode": 0.003,
      "not_slide_no_signal": 0.001,
      "may_be_slide_powerpoint_side_screen": 0.000,
      "not_slide_black_or_blue_screen": 0.000
    }
  }
]
```

## Programmatic Usage

You can also use the classifier programmatically in your Node.js applications:

```javascript
const { SlideClassifier } = require('./inference.js');

async function classifyImage() {
    const classifier = new SlideClassifier(
        'models/slide_classifier_mobilenetv4.onnx',
        'models/model_info.json',
        0.5 // confidence threshold
    );

    await classifier.loadModel();

    // Classify single image
    const result = await classifier.predictSingle('path/to/image.png', true);
    console.log('Prediction:', result.predicted_class);
    console.log('Confidence:', result.confidence);

    // Classify multiple images
    const results = await classifier.predictBatch([
        'image1.png',
        'image2.png'
    ], true);

    results.forEach(result => {
        console.log(`${result.image_path}: ${result.predicted_class} (${result.confidence.toFixed(3)})`);
    });
}

classifyImage().catch(console.error);
```

## Performance

- **Preprocessing**: Uses Sharp for fast image processing
- **Inference**: ONNX Runtime provides optimized inference
- **Memory**: Efficient memory usage with streaming image processing
- **Speed**: Typically 2-5x faster than PyTorch for single image inference

## Troubleshooting

### Installation Issues

**Sharp installation problems:**
```bash
# Clear npm cache and reinstall
npm cache clean --force
npm install sharp --verbose
```

**ONNX Runtime issues:**
```bash
# Try installing specific version
npm install onnxruntime-node@1.16.3
```

### Runtime Issues

**Memory errors with large images:**
- The script automatically resizes images to 256x256
- If still having issues, process images in smaller batches

**Model loading errors:**
- Ensure the ONNX model file exists: `models/slide_classifier_mobilenetv4.onnx`
- Ensure the model info file exists: `models/model_info.json`
- Check file permissions

**Image processing errors:**
- Supported formats: JPG, PNG, BMP, TIFF, WebP
- Ensure images are not corrupted
- Check file permissions

## Comparison with Python Version

| Feature | Python (inference.py) | Node.js (inference.js) |
|---------|----------------------|------------------------|
| Model Format | PyTorch (.pth) | ONNX (.onnx) |
| Dependencies | PyTorch, timm, PIL | ONNX Runtime, Sharp |
| Performance | Good | Faster (2-5x) |
| Memory Usage | Higher | Lower |
| Deployment | Requires Python env | Standalone Node.js |
| Cross-platform | Yes | Yes |

## Integration Examples

### Express.js API

```javascript
const express = require('express');
const multer = require('multer');
const { SlideClassifier } = require('./inference.js');

const app = express();
const upload = multer({ dest: 'uploads/' });
const classifier = new SlideClassifier(
    'models/slide_classifier_mobilenetv4.onnx',
    'models/model_info.json'
);

app.post('/classify', upload.single('image'), async (req, res) => {
    try {
        const result = await classifier.predictSingle(req.file.path);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

classifier.loadModel().then(() => {
    app.listen(3000, () => console.log('Server running on port 3000'));
});
```

### Batch Processing Script

```javascript
const fs = require('fs').promises;
const { SlideClassifier } = require('./inference.js');

async function batchProcess() {
    const classifier = new SlideClassifier(
        'models/slide_classifier_mobilenetv4.onnx',
        'models/model_info.json',
        0.8
    );

    await classifier.loadModel();

    const results = await classifier.predictDirectory('input_images');

    // Filter high-confidence slides
    const slides = results.filter(r =>
        r.predicted_class === 'slide' && r.confidence > 0.9
    );

    console.log(`Found ${slides.length} high-confidence slides`);

    // Save results
    await fs.writeFile('batch_results.json', JSON.stringify(results, null, 2));
}

batchProcess().catch(console.error);
```