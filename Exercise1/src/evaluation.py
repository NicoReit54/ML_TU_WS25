import pandas as pd
import uuid

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_run(classifier: str, 
                 y_true: pd.DataFrame, 
                 y_pred: pd.DataFrame,
                 id: str = "") -> dict:
    """
        Computes standard metrics and puts them into a result dict for further usage.
    
    Parameters:
    ----------
    classifier : str
        classifier that has been used for this prediction/run
    y_true : pd.DataFrame
        the dataframe containing the "true" target variables
    y_pred: pd.DataFrame
        predicted dataframe containing the target
    id: str
        A str to use as some sort of identifier for the returned dictionary (optional, if not provided a uuid4 will be added)

    Returns:
    -------
    dict
        
    """

    if id == "":
        id = str(uuid.uuid4())

    return dict(
        id=id,
        classifier=classifier,
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro"),
        precision=precision_score(y_true, y_pred, average="macro"),
        recall=recall_score(y_true, y_pred, average="macro"),
        conf_matr=confusion_matrix(y_true, y_pred)
    )

def plot_confusion_matrix(
        classifier: str, 
        y_true: pd.DataFrame, 
        y_pred: pd.DataFrame,
        figsize: tuple = (5, 5)) -> None:
    
    """

        Plots a confusion matrix based on the clasifiers predictions
    
    Parameters:
    ----------
    classifier : str
        classifier that has been used for this prediction/run
    y_true : pd.DataFrame
        the dataframe containing the "true" target variables
    y_pred: pd.DataFrame
        predicted dataframe containing the target
    figsize : tuple
        A tuple containing the wished for size of the plot

    Returns:
    -------
    None
        Displays a confusion matrix
    
    """
    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues', colorbar=False)
    disp.ax_.set_title(f"{classifier} - Confusion Matrix")

    plt.show()
    plt.tight_layout()

    return None


import matplotlib.pyplot as plt

def plot_metrics(
        result_dict: dict,
        figsize: tuple = (5,4),
        x_rotation:int = 30) -> None:
    """
        Visualizes accuracy, balanced accuracy, F1, precision, and recall as a bar chart.
    
    Parameters:
    -------
    result_dict: dict
        Contains the results (such as accuracy and f1) produced by evaluate_run()
    x_rotation: int 
        Degrees in how the x-axis labels shall be set
    figsize : tuple
        A tuple containing the wished for size of the plot

    Returns:
    -------
    None
        Plots the metrics
    """
    metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    values = [result_dict[m] for m in metrics]

    plt.figure(figsize=figsize)

    bars = plt.bar(metrics, values, color='skyblue', edgecolor='black')

    plt.ylim(0, 1)
    plt.title(f"Performance Metrics — {result_dict['classifier']}")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tick_params(axis="x", labelrotation = x_rotation)

    # Add values to the bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return None