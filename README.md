# ResNet-18 CIFAR-10 Training

This repository contains training code for fine-tuning ResNet-18 on CIFAR-10
using Red Hat OpenShift AI Training Operator.

## Usage

This code is designed to run as a PyTorchJob on RHOAI. The training script
accepts configuration through environment variables:

- \`EPOCHS\`: Number of training epochs (default: 10)
- \`BATCH_SIZE\`: Training batch size (default: 32)
- \`LEARNING_RATE\`: Learning rate (default: 0.001)
- \`DATA_DIR\`: Directory for CIFAR-10 data (default: /shared/data)
- \`OUTPUT_DIR\`: Directory for model outputs (default: /shared/models)

## Training Features

- Automatic dataset download
- Checkpoint management with recovery
- TensorBoard logging
- Best model saving
- Training history tracking