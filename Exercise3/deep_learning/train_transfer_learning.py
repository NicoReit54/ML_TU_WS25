# SOURCES:
# https://medium.com/@draj0718/image-classification-and-prediction-using-transfer-learning-3cf2c736589d
# https://www.geeksforgeeks.org/deep-learning/multiclass-image-classification-using-transfer-learning/
# https://www.researchgate.net/publication/325803364_A_Study_on_CNN_Transfer_Learning_for_Image_Classification
# https://www.tensorflow.org/tutorials/images/transfer_learning#data_preprocessing
# https://medium.com/data-science/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2
# https://www.kaggle.com/code/gauthamupadhyaya/image-classification-using-cnn-and-efficientnetb0
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from data.load_cifar import CIFAR10
from data.load_gtsrb import GTSRB, GTSRB_CLASSES
from analysis.evaluation import evaluate_model
from cnn_transfer_learning import EfficientNetTransferModel


# ============================================================================
# DATA AUGMENTATION & PREPARATION
# ============================================================================

def get_cifar_augmentation():
    """
    Get data augmentation transforms for CIFAR-10.
    Includes resizing to 224x224 for EfficientNet.
    """
    return transforms.Compose([
        transforms.Resize(224),  # Upsample to EfficientNet input size
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])


def get_gtsrb_augmentation():
    """
    Get data augmentation transforms for GTSRB (traffic signs).
    Includes rotation and affine transforms suitable for traffic signs.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])


def get_test_transform():
    """Get transform for test data (no augmentation, just resize)"""
    return transforms.Resize(224)


def prepare_cifar10_loaders(batch_size=64, num_workers=2):
    """
    Prepare CIFAR-10 data loaders using existing CIFAR10 class.
    
    :param batch_size: Batch size for training and testing
    :param num_workers: Number of workers for data loading
    :return: (train_loader, test_loader, class_names)
    """
    train_transform = get_cifar_augmentation()
    test_transform = get_test_transform()
    
    train_ds = CIFAR10(split="train", transform=train_transform)
    test_ds = CIFAR10(split="test", transform=test_transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, test_loader, class_names


def prepare_gtsrb_loaders(data_dir, batch_size=64, num_workers=2, sample_fraction=1.0):
    """
    Prepare GTSRB data loaders using existing GTSRB class.
    
    :param data_dir: Base directory containing GTSRB data
    :param batch_size: Batch size for training and testing
    :param num_workers: Number of workers for data loading
    :param sample_fraction: Fraction of data to use (for debugging)
    :return: (train_loader, test_loader, class_names)
    """
    train_path = os.path.join(data_dir, "Final_Training/Images")
    test_path = os.path.join(data_dir, "Final_Test/Images")
    
    train_transform = get_gtsrb_augmentation()
    test_transform = get_test_transform()
    
    train_ds = GTSRB(
        base_path=train_path, split="train", 
        sample_fraction=sample_fraction, transform=train_transform
    )
    test_ds = GTSRB(
        base_path=test_path, split="test", 
        sample_fraction=sample_fraction, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    class_names = [GTSRB_CLASSES[i] for i in range(43)]
    
    return train_loader, test_loader, class_names


# ============================================================================
# TRAINING FUNCTIONS (adapted from train.py)
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    Adapted from train.py.
    
    :return: (epoch_loss, epoch_accuracy)
    """
    running_loss = 0
    correct = 0
    total = 0
    
    model.train()
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate model on test/validation set.
    Adapted from train.py.
    
    :return: (epoch_loss, epoch_accuracy)
    """
    running_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_transfer_learning(
    model, 
    train_loader, 
    test_loader, 
    device,
    epochs_frozen=5, 
    epochs_unfrozen=10,
    lr_frozen=1e-3, 
    lr_unfrozen=1e-4,
    save_path='best_model.pth',
    plot_dir='transfer_learning_curves'
):
    """
    Two-stage transfer learning training pipeline:
    1. Stage 1: Train classifier head with frozen backbone
    2. Stage 2: Fine-tune last layers with lower learning rate
    
    Adapted from train.py structure with timing and plotting.
    
    :param model: The transfer learning model
    :param train_loader: Training data loader
    :param test_loader: Test/validation data loader
    :param device: Device to train on (cuda/cpu/mps)
    :param epochs_frozen: Number of epochs for stage 1
    :param epochs_unfrozen: Number of epochs for stage 2
    :param lr_frozen: Learning rate for stage 1
    :param lr_unfrozen: Learning rate for stage 2
    :param save_path: Path to save best model
    :param plot_dir: Directory to save training curves
    :return: (trained_model, best_accuracy, history_dict)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    total_train_time = 0.0
    total_val_time = 0.0
    
    # ========== STAGE 1: Train classifier head ==========
    print("\n" + "="*60)
    print("STAGE 1: Training classifier head (frozen backbone)")
    print("="*60)
    
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr_frozen
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_frozen)
    
    best_acc = 0
    for epoch in range(epochs_frozen):
        print(f"\nEpoch {epoch+1}/{epochs_frozen}")
        
        # Training with timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        train_time = time.perf_counter() - t0
        total_train_time += train_time
        
        # Validation with timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        val_time = time.perf_counter() - t0
        total_val_time += val_time
        
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Time: {val_time:.2f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Val Acc: {best_acc:.4f}")
        
        best_acc = max(val_accuracies)
        
    # ========== STAGE 2: Fine-tune last layers ==========
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning last layers")
    print("="*60)
    
    model.unfreeze_backbone(num_layers=2)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr_unfrozen
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_unfrozen)
    
    best_acc = max(val_accuracies)
    
    for epoch in range(epochs_unfrozen):
        print(f"\nEpoch {epochs_frozen + epoch + 1}/{epochs_frozen + epochs_unfrozen}")
        
        # Training with timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        train_time = time.perf_counter() - t0
        total_train_time += train_time
        
        # Validation with timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        val_time = time.perf_counter() - t0
        total_val_time += val_time
        
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Time: {val_time:.2f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Val Acc: {best_acc:.4f}")
    
    # ========== Plot results ==========
    print(f"\nTotal Training Time  : {total_train_time:.2f}s")
    print(f"Total Validation Time: {total_val_time:.2f}s")
    
    plot_training_curves(
        train_losses, val_losses, 
        train_accuracies, val_accuracies,
        epochs_frozen, plot_dir
    )
    
    # Return history for notebook analysis
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'total_train_time': total_train_time,
        'total_val_time': total_val_time
    }
    
    return model, best_acc, history


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         freeze_point, save_dir='transfer_learning_curves', 
                         model_name='EfficientNetB0'):
    """
    Plot training curves with stage separation.
    Adapted from train.py
    
    :param freeze_point: Epoch where backbone was unfrozen
    :param model_name: Name to use in filename for unique identification
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.axvline(x=freeze_point, color='gray', linestyle='--', 
                alpha=0.5, label='Backbone Unfrozen')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Accuracy")
    plt.plot(epochs_range, val_accs, label="Validation Accuracy")
    plt.axvline(x=freeze_point, color='gray', linestyle='--', 
                alpha=0.5, label='Backbone Unfrozen')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create unique filename
    import time
    timestamp = int(time.time())
    file_name = f"{save_dir}/training_curves_{model_name}_{timestamp}.png"
    
    try:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {file_name}")
    except Exception as e:
        print(f"Could not save the figure: {e}")
    
    plt.show()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Transfer Learning with EfficientNet-B0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cifar10', 'gtsrb'],
                       help='Dataset to use for training')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (used for GTSRB path)')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                       help='Fraction of GTSRB data to use (1.0 = all data)')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for classifier head')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from scratch without pretrained weights')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs-frozen', type=int, default=5,
                       help='Epochs to train with frozen backbone')
    parser.add_argument('--epochs-unfrozen', type=int, default=10,
                       help='Epochs to fine-tune with unfrozen layers')
    parser.add_argument('--lr-frozen', type=float, default=1e-3,
                       help='Learning rate for frozen stage')
    parser.add_argument('--lr-unfrozen', type=float, default=1e-4,
                       help='Learning rate for unfrozen stage')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='./outputs',
                       help='Directory to save model and plots')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model filename (default: efficientnet_b0_{dataset}.pth)')
    
    # Evaluation arguments
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate a trained model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to model checkpoint to load')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cuda', 'cpu', 'mps', 'auto'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main training script - CLI entry point"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set default model name
    if args.model_name is None:
        args.model_name = f"efficientnet_b0_{args.dataset}.pth"
    model_path = os.path.join(args.save_dir, args.model_name)
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} dataset...")
    if args.dataset == 'cifar10':
        train_loader, test_loader, class_names = prepare_cifar10_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        num_classes = 10
    else:  # gtsrb
        train_loader, test_loader, class_names = prepare_gtsrb_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_fraction=args.sample_fraction
        )
        num_classes = 43
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating EfficientNet-B0 model for {num_classes} classes...")
    model = EfficientNetTransferModel(
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=args.pretrained
    )
    
    # Load checkpoint if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    
    # Print model info
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Evaluation only or training
    if args.eval_only:
        if args.load_model is None:
            raise ValueError("--eval-only requires --load-model to specify checkpoint")
        
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        # Use your existing evaluation function
        model.to(device)
        results = evaluate_model(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=str(device),
            title=f"EfficientNet-B0 on {args.dataset.upper()}"
        )
        
        print(f"\nFinal Test Accuracy: {results['overall_accuracy']:.4f}")
        
    else:
        # Train model
        model, best_acc, history = train_transfer_learning(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs_frozen=args.epochs_frozen,
            epochs_unfrozen=args.epochs_unfrozen,
            lr_frozen=args.lr_frozen,
            lr_unfrozen=args.lr_unfrozen,
            save_path=model_path
        )
        
        print("\n" + "="*60)
        print(f"Training complete!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print(f"Model saved to: {model_path}")
        print("="*60)
        
        # Final evaluation using your evaluation function
        print("\nRunning final evaluation on test set...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        results = evaluate_model(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=str(device),
            title=f"EfficientNet-B0 on {args.dataset.upper()}"
        )


if __name__ == "__main__":
    main()
