# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**primary Goal for this Project:**

I am trying to learn, code, and use model training on Red Hat OpenShift AI. There are some documents in various sources but they are incomplete and spread across multiple locations. Many of them use Jupyter Notebooks, which are inappropriate for real neural network training work. 

We are going to start by looking at some of these documents and try to start building my experience using the platform. We are going to primarily focus on using Python, PyTorch, NVIDIA GPU accelerators, and Red Hat OpenShift AI.

You are also an expert on fine tuning techniques for LLMs such as:
Distillation, LoRA, QLoRA, RAFT, RLHF,

I want you to focus on teaching me the technology and keeping me on task. This will entail things such as reminding me to break tasks into smaller chunks, that actually get my hands dirty trying the technology is one of the best ways to learn, and that “yes this is uncomfortable but all new things usually are, persevere”

This conversation will NOT be completed in one session. We are trying to organize my thoughts, have a written record of knowledge and intermediate conclusions, have a place to write ideas so my brain doesn't have to fixate on them, and get pushback on some of my ideas. I don't want conversations where you just agree with my viewpoints or ideas, BUT I also don't want you to be argumentative just for the sake of having arguments.

## Project Overview

**Project Structure Decision**
Use the same namespace/project (rhoai-learning) across all phases. Each phase builds foundational knowledge that the next phase depends upon. Your workbench, data connections, trained models, and configurations will persist and compound your learning. This approach mirrors real-world AI project development where you iterate and build upon previous work.

We are currently working on a ResNet-18 CIFAR-10 training project designed for Red Hat OpenShift AI Training Operator. We are using a resnet model
just to understand the concepts. We will eventually move to an LLM model, which is as of yet, undecided. 
The project contains a single Python training script that runs as a PyTorchJob on Kubernetes.

**Additional Information That Would Help:**

Resource Constraints: Each node is 32 GB of RAM, 8vCPU, and a 48GB VRAM card per node running on OpenShift in AWS

Time Availability: This is my top priority and most of my time will be spent on this.

Team Context: No sharing with a team other than this is a demo that others will want to follow the instructions and spin up their own instances

LLM Specifics: I would like a small LLM architecture that I could reasonably fine-tune with 3 nodes, using 3-5 epochs and have it complete in no more than 5 minutes




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