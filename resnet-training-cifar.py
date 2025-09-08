#!/usr/bin/env python3
"""
ResNet-18 CIFAR-10 Fine-tuning for Training Operator
Designed to run as a discrete job with proper checkpoint management
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import json


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_cifar10_data(data_dir, batch_size=32):
    """Load and prepare CIFAR-10 dataset"""
    print(f"Loading CIFAR-10 data from {data_dir}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Standard transform for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download/load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Classes: {classes}")

    return trainloader, testloader, classes


def create_model(num_classes=10):
    """Create ResNet-18 model for CIFAR-10"""
    model = torchvision.models.resnet18(pretrained=True)

    # Modify for CIFAR-10 (smaller input size)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for 32x32 images

    # Update classifier for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_checkpoint(model, optimizer, epoch, loss, output_dir):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Also save as 'latest' for easy recovery
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']


def train_epoch(model, device, trainloader, optimizer, criterion, epoch, writer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log every 100 batches
        if batch_idx % 100 == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(trainloader)}, '
                  f'Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%')

            # TensorBoard logging
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Batch_Accuracy', batch_acc, global_step)

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, device, testloader, criterion):
    """Validate model performance"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def main():
    """Main training function"""
    print("Starting ResNet-18 CIFAR-10 Training")
    print("=" * 50)

    # Get configuration from environment variables
    epochs = int(os.getenv('EPOCHS', '10'))
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    learning_rate = float(os.getenv('LEARNING_RATE', '0.001'))
    data_dir = os.getenv('DATA_DIR', '/shared/data')
    output_dir = os.getenv('OUTPUT_DIR', '/shared/models')

    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Data Directory: {data_dir}")
    print(f"  Output Directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = get_device()

    # Load data
    trainloader, testloader, classes = load_cifar10_data(data_dir, batch_size)

    # Create model
    model = create_model(num_classes=len(classes))
    model = model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup TensorBoard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    best_acc = 0
    training_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_epoch(model, device, trainloader, optimizer, criterion, epoch, writer)

        # Validate
        val_loss, val_acc = validate(model, device, testloader, criterion)

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # Print epoch results
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, output_dir)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")

        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Save final model and history
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save training summary
    summary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_accuracy': best_acc,
        'total_time': total_time,
        'device': str(device)
    }

    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    writer.close()
    print(f"Training artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()