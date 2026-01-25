import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from analysis.evaluation import evaluate_model
from deep_learning.cnn_transfer_learning import EfficientNetTransferModel
from deep_learning.train_transfer_learning import *

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
        
        # Use existing evaluation function
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
