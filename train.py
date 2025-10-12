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
import argparse
from pathlib import Path

# Enable Metal Performance Shaders (MPS) for M4 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss, model):
        """
        Call this method after each epoch

        Args:
            val_loss (float): Current validation loss
            model: The model being trained

        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    print(f"Restoring model weights from epoch with best validation loss: {self.best_loss:.4f}")
                model.load_state_dict(self.best_weights)
            return True

        return False

    def get_best_loss(self):
        return self.best_loss

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

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint for fine-tuning

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state

    Returns:
        dict: Checkpoint information (epoch, best_val_acc, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state")

    start_epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)

    print(f"Resuming from epoch {start_epoch}, best validation accuracy: {best_val_acc:.2f}%")

    return {
        'start_epoch': start_epoch,
        'best_val_acc': best_val_acc,
        'class_names': checkpoint.get('class_names', []),
        'class_to_idx': checkpoint.get('class_to_idx', {}),
        'config': checkpoint.get('config', {})
    }

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train slide classifier with MobileNetV4')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='slide_classifier_mobilenetv4.pth',
                        help='Model filename')
    parser.add_argument('--force_fresh', action='store_true',
                        help='Force fresh training without prompting to resume from existing model')

    args = parser.parse_args()

    # Configuration
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'save_dir': args.save_dir,
        'model_name': args.model_name,
        'resume_from': args.resume,
        'force_fresh': args.force_fresh
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

    # Initialize training variables
    start_epoch = 0
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Load checkpoint if resuming training
    if config['resume_from'] is not None:
        checkpoint_info = load_checkpoint(
            config['resume_from'], model, optimizer, scheduler
        )
        start_epoch = checkpoint_info['start_epoch']
        best_val_acc = checkpoint_info['best_val_acc']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        # Check if a previous best model exists and offer to resume
        model_path = os.path.join(config['save_dir'], config['model_name'])
        if os.path.exists(model_path) and not config['force_fresh']:
            print(f"\nFound existing model: {model_path}")
            try:
                # Try to load checkpoint info to show details
                checkpoint = torch.load(model_path, map_location=device)
                prev_epoch = checkpoint.get('epoch', 'unknown')
                prev_acc = checkpoint.get('best_val_acc', 'unknown')
                print(f"Previous training: Epoch {prev_epoch}, Best Val Acc: {prev_acc:.2f}%")

                response = input("Do you want to resume from this model? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    checkpoint_info = load_checkpoint(
                        model_path, model, optimizer, scheduler
                    )
                    start_epoch = checkpoint_info['start_epoch']
                    best_val_acc = checkpoint_info['best_val_acc']
                    print(f"Resuming training from epoch {start_epoch}")
                else:
                    print("Starting fresh training (existing model will be overwritten)")
            except Exception as e:
                print(f"Could not read existing model file: {e}")
                print("Starting fresh training")
        elif os.path.exists(model_path) and config['force_fresh']:
            print(f"Found existing model but --force_fresh specified. Starting fresh training.")
            print(f"Existing model will be overwritten: {model_path}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=0.001,
        restore_best_weights=True,
        verbose=True
    )

    print(f"\nStarting training on {device}...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")

    for epoch in range(start_epoch, config['num_epochs']):
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

        # Check early stopping
        if early_stopping(val_loss, model):
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            print(f'Best validation loss: {early_stopping.get_best_loss():.4f}')
            break

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
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': full_dataset.classes,
                'class_to_idx': full_dataset.class_to_idx,
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
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