import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    figsize: tuple = (10, 8),
    device: str = "cpu",
    title="Model Evaluation") -> dict[str, Any]:
    """
    Includes:
    - Overall accuracy
    - Per-class accuracy
    - Precision / Recall / F1-score
    - Confusion matrix visualization

    Returns all metrics in a dictionary for comparison.
    """

    # Set to evaluation mode to disable e.g. dropout
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Overall accuracy
    overall_accuracy = (all_preds == all_labels).float().mean().item()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Print metrics
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Overall accuracy: {overall_accuracy:.4f}\n")

    print("Per-class accuracy:")
    for idx, acc in enumerate(per_class_accuracy):
        print(f"{class_names[idx]:>10}: {acc:.3f}")

    # precision, recall, f1-score via classification report
    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=3
    ))

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix – {title}")
    plt.tight_layout()
    plt.show()

    # Return metrics for comparison of multiple models
    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels
    }
