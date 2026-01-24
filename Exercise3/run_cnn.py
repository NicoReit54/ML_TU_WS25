"""
CNN Training and Classification Tool for GTSRB and CIFAR-10 Datasets
This module provides a complete pipeline for training our custom SimpleCNN model
on either the GTSRB (German Traffic Sign Recognition Benchmark) or CIFAR-10 dataset.

Supported Command-Line Arguments:
    --dataset: 'GTSRB' or 'CIFAR' (required)
    --epochs: Number of training epochs (default: 10)
    --channel_multiplier: Channel growth factor per conv layer (default: 2)
    --base_channels: Initial channels in first conv layer (default: 32)
    --dropout_rate: Dropout rate for regularization (default: 0.5)
    --kernel_size: Convolution kernel size (default: 3)
    --padding: Padding for convolutions (default: 1)
    --batch_size: Batch size for dataloaders (default: 128)
    --data_augmentation: Enable/disable augmentation (default: True)
    --validation: Enable/disable validation output (default: True)

"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from deep_learning.cnn import SimpleCNN # TODO: Change SimpleCNN to accept the same as TestCNN
from deep_learning.train import train_model

from data.load_cifar import CIFAR10
from data.load_gtsrb import GTSRB_CLASSES, GTSRB

from analysis.evaluation import evaluate_model

def load_data(data_set, data_augmentation, batch_size=128) -> tuple[DataLoader, DataLoader, torch.device]:
    '''
    Loads the specified dataset (GTSRB or CIFAR-10), applies data augmentation if specified, 
    and returns the training and validation dataloaders along with the automatically detected device (GPU/CPU).    

    :param data_set: Dataset to load ('GTSRB' or 'CIFAR').
    :type data_set: str
    :param data_augmentation: Whether to apply data augmentation.
    :type data_augmentation: bool
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :return: Training and validation dataloaders along with the device.
    :rtype: tuple[DataLoader, DataLoader, device]
    '''
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data Augmentation
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(64, padding=4), # padding is ofc applied before cropping randomly
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    ])
    if data_augmentation:
        train_transforms = train_transforms
    else:
        train_transforms = None

    # Create datasets
    if data_set == 'GTSRB':
        training_path = "data/GTSRB/Final_Training/Images"
        testing_path = "data/GTSRB/Final_Test/Images"

        print("Loading GTSRB training dataset...")
        train_dataset = GTSRB(base_path=training_path, split="train", transform=train_transforms)
        print("Loading GTSRB testing dataset...")
        test_dataset  = GTSRB(base_path=testing_path, split="test", transform=None)

    elif data_set == 'CIFAR':
        print("Loading CIFAR-10 training dataset...")
        train_dataset = CIFAR10(split="train", transform=train_transforms)
        print("Loading CIFAR-10 testing dataset...")
        test_dataset  = CIFAR10(split="test", transform=None)   

    else:
        raise ValueError("Dataset not recognized. Use 'GTSRB' or 'CIFAR'.")
    
    # Create dataloaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    valloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # check the "batching" if it works as intended
    images, labels = next(iter(trainloader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    return trainloader, valloader, device


def main():
    '''
    Main function to parse the arguments, set up the model accordingly, and initiate training (and evaluation if wanted).
    '''
    parser = argparse.ArgumentParser(description="CNN Training and Classification Tool")

    # Configuration Flags ---
    parser.add_argument("--dataset", type=str, required=True, choices=['GTSRB', 'CIFAR'], help="Dataset to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run this on.")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="Factor by which the number of channels increases after each convolutional layer.")
    parser.add_argument("--base_channels", type=int, default=32, help="Number of wanted channels in the first convolutional layer.")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the dropout layer to reduce overfitting.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel Size")
    parser.add_argument("--padding", type=int, default=1, help="Padding.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    parser.add_argument("--data_augmentation", type=bool, default=True, help="True/False of whether data_augmentation shall be used")
    parser.add_argument("--validation", type=bool, default=True, help="True/False of whether validation output should be produced (conf. matrix, ...)")

    args = parser.parse_args()

    # classic rgb variable
    in_channels = 3
    
    # parsed variables
    epochs = int(args.epochs)
    channel_multiplier = int(args.channel_multiplier)
    base_channels = int(args.base_channels)
    dropout_rate = float(args.dropout_rate)
    kernel_size = int(args.kernel_size)
    padding = int(args.padding)
    batch_size = int(args.batch_size)
    data_augmentation = bool(args.data_augmentation)
    validation = bool(args.validation)

    # set appropriate input_size and load the data
    if args.dataset == 'GTSRB':
        input_size = 64
        num_classes = len(GTSRB_CLASSES) # > Number of available output classes for classification.
    elif args.dataset == 'CIFAR':
        input_size = 32
        num_classes = 10
    
    trainloader, valloader, device = load_data(args.dataset, data_augmentation, batch_size=batch_size)

    model = SimpleCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_multiplier=channel_multiplier,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        padding=padding,
        input_size=input_size
    )
    model_name = f"SimpleCNN_{args.dataset}_bs{batch_size}_ep{epochs}_cm{channel_multiplier}_bc{base_channels}_dr{dropout_rate}_ks{kernel_size}_pad{padding}"
    
    print(f"Training model: {model_name}")
    # Set the model to the device (GPU/CPU)
    model = model.to(device)
    print(model)

    # Prepare Loss function and optimizer
    criterion = nn.CrossEntropyLoss() # loss function cross entropy that will be used for calculating the loss during training so that the optimizer can update the weights accordingly (remember backpropagation!)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    print("Starting training...")
    # Train the model
    trained_model, train_accuracies, val_accuracies = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs
        )
    print("Training completed.")    
    print("Final Train Accuracy:", train_accuracies[-1])
    print("Final Val   Accuracy:", val_accuracies[-1])

    print("\nTrain Accuracies:", train_accuracies)
    print("Val   Accuracies:", val_accuracies)

    print("\nSaving trained model...")
    # Save the trained model and check for exisiting files/dirs 
    Path("cnn_trained_models").mkdir(exist_ok=True)
    file_name = f"cnn_trained_models/{model_name.lower()}.pth"
    
    # Check if file exists and append version number
    base_path = Path(file_name)
    if base_path.exists():
        version = 1
        while (base_path.parent / f"{base_path.stem}_v{version}{base_path.suffix}").exists():
            version += 1
        file_name = str(base_path.parent / f"{base_path.stem}_v{version}{base_path.suffix}")

    torch.save(trained_model.state_dict(), file_name)
    print("Saved trained model to:", file_name)
    
    if validation:
        if args.dataset == 'GTSRB':
            class_names = GTSRB_CLASSES
        elif args.dataset == 'CIFAR':
            class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
       
        print("Evaluating trained model on validation set...")
        evaluate_model(trained_model, valloader, device=device, class_names=class_names)

    del trained_model

if __name__ == "__main__":
    main()
