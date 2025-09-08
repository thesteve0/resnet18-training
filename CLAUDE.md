# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ResNet-18 CIFAR-10 training project designed for Red Hat OpenShift AI Training Operator. The project contains a single Python training script that runs as a PyTorchJob on Kubernetes.

## Key Files

- `resnet-training-cifar.py`: Main training script with ResNet-18 implementation for CIFAR-10
- `requirements.txt`: Python dependencies (PyTorch, TorchVision, TensorBoard)
- `deploy/pytorch-training-job.yaml`: Kubernetes PyTorchJob definition for RHOAI
- `deploy/pvcbindings.yaml`: Persistent Volume Claims for data, models, and workspace storage

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run training locally (uses environment variables for configuration)
python resnet-training-cifar.py
```

### Environment Variables
The training script accepts configuration through environment variables:
- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Training batch size (default: 32) 
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `DATA_DIR`: Directory for CIFAR-10 data (default: /shared/data)
- `OUTPUT_DIR`: Directory for model outputs (default: /shared/models)

### Kubernetes Deployment
```bash
# Deploy PVCs first
kubectl apply -f deploy/pvcbindings.yaml

# Deploy training job
kubectl apply -f deploy/pytorch-training-job.yaml

# Monitor job progress
kubectl logs -f pytorchjob/resnet18-cifar10-git

# Check job status
kubectl get pytorchjobs
```

## Architecture

### Training Pipeline
The training script (`resnet-training-cifar.py`) implements:

1. **Model Architecture**: ResNet-18 adapted for CIFAR-10 (32x32 images)
   - Modified conv1 layer (3x3 kernel, stride=1, padding=1)
   - Removed maxpool layer for smaller input size
   - 10-class output layer for CIFAR-10

2. **Data Pipeline**: CIFAR-10 with data augmentation
   - Training: RandomCrop, RandomHorizontalFlip, normalization
   - Validation: Standard normalization only

3. **Training Features**:
   - Automatic checkpoint saving/loading
   - TensorBoard logging 
   - Best model tracking
   - Training history JSON export
   - GPU/CPU automatic detection

4. **Output Artifacts** (saved to `OUTPUT_DIR`):
   - `checkpoint_latest.pth`: Latest training checkpoint
   - `best_model.pth`: Best performing model
   - `final_model.pth`: Final trained model
   - `training_history.json`: Epoch-by-epoch metrics
   - `training_summary.json`: Training configuration and results
   - `tensorboard/`: TensorBoard logs

### Kubernetes Architecture
- **PyTorchJob**: Single master replica with GPU allocation
- **Init Container**: Git clone for code deployment
- **Storage**: Three PVCs for data, models, and workspace
- **Resource Allocation**: 4-6 CPU, 16-24Gi memory, 1 GPU

## Testing

No formal test suite is included. The training script includes validation during training and saves metrics for evaluation.