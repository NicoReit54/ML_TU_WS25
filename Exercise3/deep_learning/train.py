import time
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Sources: 
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html 
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# and CoPilot (esp. for fancy tqdm! )


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    running_loss = 0
    correct = 0
    total = 0

    # Set the model to training mode, enabling dropout (to improve generalization and avoid overfitting), batch normalization updates...
    model.train()
    
    # Iterate over the entire training dataset in this one batch
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        
        # Move data to GPU/CPU depending on the selected "device" (weird name for that)
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients for every batch: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        # (why? > In Pytorch gradients get accumulated by default due to some convenient
        # handling for other models. If not set Otherwise, the gradient would be a combination of the old gradient, 
        # which we would have already used to update our model parameters and the newly-computed gradient. 
        # It would therefore point in some other direction than the intended direction towards the minimum)
        optimizer.zero_grad()

        # Making prediction for each batch
        outputs = model(images)
        
        # Computing loss and its gradients 
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjusting the learning weights, i.e. updating the model parameters (gradient descent step)
        optimizer.step()

        # accumulate the loss for statistics (loss.item() converts the tensor into a float > funfact: moves it into the CPU if before GPU)
        running_loss += loss.item() * images.size(0) # times batch size to get the entire loss per epoch as .item() gives the mean loss

        # Compute the number of correct predictions in this batch for stats later
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    # Compute average loss and accuracy for the entire epoch
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    running_loss = 0
    correct = 0
    total = 0

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation to speed up validation (and reduce memory usage)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            # compute outputs based on the current model and validation images
            outputs = model(images)
            
            # compute loss
            loss = criterion(outputs, labels)

            # get the stats again
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Combining the two above now
def train_model(model, trainloader, valloader, optimizer, criterion, device, epochs=10):
    model.to(device) # again CPU/GPU decision
    total_train_time, total_val_time = 0.0, 0.0 # initialize "time stamps"

    # For the plotting later
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training + timing
        torch.cuda.synchronize() if device.type == "cuda" else None # just necessary so we dont measure waiting times for the GPU only
        t0 = time.perf_counter()
        
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        train_time = t1 - t0
        total_train_time += train_time

        # Validation + timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        val_loss, val_acc = validate(model, valloader, criterion, device)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        val_time = t1 - t0
        total_val_time += val_time

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Time: {val_time:.2f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)


    # Output of all the stats for the entire training process + plotting
    print(f"\nTotal Training Time  : {total_train_time:.2f}s")
    print(f"Total Validation Time: {total_val_time:.2f}s")

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Check if file and dir exist and also append version number
    Path("cnn_training_curves").mkdir(exist_ok=True)

    file_name = f"cnn_training_curves/training_curves_{model.__class__.__name__}.png"

    base_path = Path(file_name)
    if base_path.exists():
        version = 1
        while (base_path.parent / f"{base_path.stem}_v{version}{base_path.suffix}").exists():
            version += 1
        file_name = str(base_path.parent / f"{base_path.stem}_v{version}{base_path.suffix}")
    
    # Try/catch for saving the figure because its more important to have the model than the plot
    try:
        plt.savefig(file_name, dpi=300)
    except Exception as e:
        print(f"Could not save the figure: {e}")
    
    plt.show()

    return model
