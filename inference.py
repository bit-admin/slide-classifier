#!/usr/bin/env python3
"""
Slide Classification Inference Script
Uses trained MobileNetV4 model to classify slide images
"""

import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
from pathlib import Path
import json
import time
from collections import defaultdict

# Enable Metal Performance Shaders (MPS) for M4 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SlideClassifier:
    """Slide classification inference class"""

    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the classifier

        Args:
            model_path (str): Path to the trained model checkpoint
            confidence_threshold (float): Minimum confidence for predictions
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        self.class_to_idx = None
        self.transform = None

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """Load the trained model and metadata"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=device)

        # Extract metadata
        self.class_names = checkpoint.get('class_names', [])
        self.class_to_idx = checkpoint.get('class_to_idx', {})
        num_classes = len(self.class_names)

        if not self.class_names:
            raise ValueError("No class names found in checkpoint")

        print(f"Found {num_classes} classes: {self.class_names}")

        # Create model architecture
        self.model = timm.create_model(
            "hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k",
            pretrained=False,
            num_classes=num_classes
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        # Print model info
        best_val_acc = checkpoint.get('best_val_acc', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Model trained for {epoch} epochs, best validation accuracy: {best_val_acc:.2f}%")

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_single(self, image_path, return_probabilities=False):
        """
        Predict class for a single image

        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return class probabilities

        Returns:
            dict: Prediction results with class, confidence, and optionally probabilities
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()

                result = {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'above_threshold': confidence_score >= self.confidence_threshold
                }

                if return_probabilities:
                    class_probs = {}
                    for i, class_name in enumerate(self.class_names):
                        class_probs[class_name] = probabilities[0][i].item()
                    result['class_probabilities'] = class_probs

                return result

        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0,
                'above_threshold': False
            }

    def predict_batch(self, image_paths, return_probabilities=False, show_progress=True):
        """
        Predict classes for multiple images

        Args:
            image_paths (list): List of image file paths
            return_probabilities (bool): Whether to return class probabilities
            show_progress (bool): Whether to show progress

        Returns:
            list: List of prediction results
        """
        results = []
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            if show_progress and i % 10 == 0:
                print(f"Processing image {i+1}/{total_images}: {os.path.basename(image_path)}")

            result = self.predict_single(image_path, return_probabilities)
            results.append(result)

        return results

    def predict_directory(self, directory_path, return_probabilities=False, show_progress=True):
        """
        Predict classes for all images in a directory

        Args:
            directory_path (str): Path to directory containing images
            return_probabilities (bool): Whether to return class probabilities
            show_progress (bool): Whether to show progress

        Returns:
            list: List of prediction results
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = [str(path) for path in sorted(image_paths)]

        if not image_paths:
            print(f"No images found in {directory_path}")
            return []

        print(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths, return_probabilities, show_progress)

def print_results_summary(results):
    """Print a summary of prediction results"""
    if not results:
        print("No results to summarize")
        return

    # Count predictions by class
    class_counts = defaultdict(int)
    confidence_scores = []
    errors = 0
    above_threshold = 0

    for result in results:
        if 'error' in result:
            errors += 1
        else:
            class_counts[result['predicted_class']] += 1
            confidence_scores.append(result['confidence'])
            if result['above_threshold']:
                above_threshold += 1

    total_processed = len(results)
    successful = total_processed - errors

    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_processed}")
    print(f"Successful predictions: {successful}")
    print(f"Errors: {errors}")
    print(f"Above confidence threshold: {above_threshold}/{successful}")

    if confidence_scores:
        avg_confidence = np.mean(confidence_scores)
        print(f"Average confidence: {avg_confidence:.3f}")

    print("\nClass Distribution:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / successful) * 100 if successful > 0 else 0
        print(f"{class_name:35} {count:4d} ({percentage:5.1f}%)")

def print_detailed_results(results, show_probabilities=False, max_results=None):
    """Print detailed prediction results"""
    if not results:
        print("No results to display")
        return

    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    display_results = results[:max_results] if max_results else results

    for i, result in enumerate(display_results, 1):
        image_name = os.path.basename(result['image_path'])

        if 'error' in result:
            print(f"{i:3d}. {image_name}")
            print(f"     ERROR: {result['error']}")
        else:
            confidence_indicator = "✓" if result['above_threshold'] else "✗"
            print(f"{i:3d}. {image_name}")
            print(f"     Class: {result['predicted_class']}")
            print(f"     Confidence: {result['confidence']:.3f} {confidence_indicator}")

            if show_probabilities and 'class_probabilities' in result:
                print("     All probabilities:")
                sorted_probs = sorted(result['class_probabilities'].items(),
                                    key=lambda x: x[1], reverse=True)
                for class_name, prob in sorted_probs:
                    print(f"       {class_name:30} {prob:.3f}")

        print()

    if max_results and len(results) > max_results:
        print(f"... and {len(results) - max_results} more results")

def save_results(results, output_path):
    """Save results to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Slide Classification Inference')
    parser.add_argument('--model', type=str, default='models/slide_classifier_mobilenetv4.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, default='test',
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON file')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence threshold for predictions')
    parser.add_argument('--show_probabilities', action='store_true',
                        help='Show probabilities for all classes')
    parser.add_argument('--max_display', type=int, default=20,
                        help='Maximum number of detailed results to display (0 for all)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only show summary, not detailed results')

    args = parser.parse_args()

    # Initialize classifier
    try:
        classifier = SlideClassifier(args.model, args.confidence_threshold)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine input type and make predictions
    input_path = Path(args.input)
    start_time = time.time()

    if input_path.is_file():
        print(f"Processing single image: {args.input}")
        results = [classifier.predict_single(str(input_path), args.show_probabilities)]
    elif input_path.is_dir():
        print(f"Processing directory: {args.input}")
        results = classifier.predict_directory(str(input_path), args.show_probabilities)
    else:
        print(f"Input path not found: {args.input}")
        return

    processing_time = time.time() - start_time

    if not results:
        print("No images processed")
        return

    # Display results
    print_results_summary(results)

    if not args.quiet:
        max_display = args.max_display if args.max_display > 0 else None
        print_detailed_results(results, args.show_probabilities, max_display)

    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/len(results):.3f} seconds")

    # Save results if requested
    if args.output:
        save_results(results, args.output)

if __name__ == '__main__':
    main()