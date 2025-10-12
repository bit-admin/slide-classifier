#!/usr/bin/env python3
"""
Slide Classification Training Script
Optimized for M4 Mac mini with Metal acceleration
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import json
import time
from pathlib import Path

# Enable Metal Performance Shaders (MPS) for M4 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SlideDataset(Dataset):
    """Custom dataset for slide classification"""

    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        # Get all classes from directory names
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load all image paths and labels
        self.samples = []
        self.class_counts = Counter()

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]

            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append((str(img_path), class_idx))
                    self.class_counts[class_idx] += 1

        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        for i, class_name in enumerate(self.classes):
            print(f"  {class_name}: {self.class_counts[i]} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load and convert image
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (256, 256), (0, 0, 0)))
            else:
                fallback = torch.zeros(3, 256, 256)
            return fallback, label

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        total_samples = len(self.samples)
        num_classes = len(self.classes)

        # Calculate weights inversely proportional to class frequency
        weights = []
        for i in range(num_classes):
            if self.class_counts[i] > 0:
                weight = total_samples / (num_classes * self.class_counts[i])
                weights.append(weight)
            else:
                weights.append(1.0)

        return torch.FloatTensor(weights)

def get_transforms():
    """Get data transforms - only color/brightness/contrast jitter as requested"""

    # Training transforms with minimal augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # MobileNetV4 expects 256x256
        transforms.ColorJitter(
            brightness=0.1,    # Slight brightness variation
            contrast=0.1,      # Slight contrast variation
            saturation=0.1,    # Slight color saturation variation
            hue=0.05          # Very slight hue variation
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform

def create_model(num_classes):
    """Create MobileNetV4 model"""
    print("Loading MobileNetV4 model...")

    # Load pretrained MobileNetV4
    model = timm.create_model(
        "hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k",
        pretrained=True,
        num_classes=num_classes
    )

    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, class_names):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    # Print detailed classification report with zero_division handling
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    return val_loss, val_acc

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.2):
    """Split dataset into train and validation sets"""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    return torch.utils.data.random_split(dataset, [train_size, val_size])

def main():
    # Configuration
    config = {
        'data_dir': 'dataset',
        'batch_size': 32,  # Reasonable for 16GB RAM
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_dir': 'models',
        'model_name': 'slide_classifier_mobilenetv4.pth'
    }

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Load dataset
    print("Loading dataset...")
    full_dataset = SlideDataset(config['data_dir'], transform=train_transform)

    # Split dataset
    train_dataset, val_dataset = split_dataset(full_dataset)

    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform

    # Calculate class weights for imbalanced dataset
    class_weights = full_dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")

    # Create data loaders - reduce num_workers to avoid "too many open files" error
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues on Mac
        pin_memory=False  # Disable pin_memory when using num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing issues on Mac
        pin_memory=False  # Disable pin_memory when using num_workers=0
    )

    # Create model
    num_classes = len(full_dataset.classes)
    model = create_model(num_classes)
    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\nStarting training on {device}...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, full_dataset.classes
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - start_time

        print(f'\nEpoch {epoch + 1}/{config["num_epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Time: {epoch_time:.2f}s')
        print('-' * 60)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(config['save_dir'], config['model_name'])

            # Save model state dict and metadata
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': full_dataset.classes,
                'class_to_idx': full_dataset.class_to_idx,
                'config': config
            }, model_path)

            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'config': config
    }

    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Model saved to: {model_path}')
    print(f'Training history saved to: {history_path}')

if __name__ == '__main__':
    main()