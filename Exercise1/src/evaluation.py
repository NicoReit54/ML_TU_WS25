import pandas as pd
import uuid
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def evaluate_run(name: str, 
                 y_true: pd.DataFrame, 
                 y_pred: pd.DataFrame,
                 task_type: Literal["classification", "regression"] = None,
                 id: str = "") -> dict:
    """
        Computes standard metrics and puts them into a result dict for further usage.
    
    Parameters:
    ----------
    name : str
        name of the classifier/model that has been used for this prediction/run
    y_true : pd.DataFrame
        the dataframe containing the "true" target variables
    y_pred: pd.DataFrame
        predicted dataframe containing the target
    task_type: str
        Either "classification" or "regression"
        If nothing is provided, error will be thrown
    id: str
        A str to use as some sort of identifier for the returned dictionary (optional, if not provided a uuid4 will be added)

    Returns:
    -------
    dict
        
    """
    # generate some ID if none is provided
    if id == "":
        id = str(uuid.uuid4())

    if task_type == "classification":
        return dict(
            id=id,
            name=name,
            task_type = task_type,
            accuracy=accuracy_score(y_true, y_pred),
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred, average="macro"),
            precision=precision_score(y_true, y_pred, average="macro"),
            recall=recall_score(y_true, y_pred, average="macro"),
            conf_matr=confusion_matrix(y_true, y_pred)
        )
    elif task_type == "regression":
        return dict(
            id=id,
            name=name,
            task_type = task_type,

            r2=r2_score(y_true, y_pred), 
            mse = mean_squared_error(y_true, y_pred), 
            mae = mean_absolute_error(y_true, y_pred), 
            mape = mean_absolute_percentage_error(y_true, y_pred)
        )
    else:
        raise ValueError(f"No valid task_type provided: {task_type}")



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

def plot_regression_results(
        y_true: pd.DataFrame, 
        y_pred: pd.DataFrame, 
        model_name: str, 
        figsize: tuple = (6,6)) -> None:
    """
        Plots the regression results as a scatterplot to compare the predicted to the test values
    
    Parameters:
    ----------
    model_name : str
        name of the model that has been used for this prediction/run
    y_true : pd.DataFrame
        the dataframe containing the "true" target variables
    y_pred: pd.DataFrame
        predicted dataframe containing the target
    figsize : tuple
        A tuple containing the wished for size of the plot

    Returns:
    -------
    None
        Plot as described
    """
    plt.figure(figsize=figsize)

    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--')
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} — Predicted vs True")
    plt.tight_layout()
    plt.show()

def plot_residuals(
        y_true: pd.DataFrame, 
        y_pred: pd.DataFrame, 
        model_name: str, 
        bins=30):

    """
        Plots the residuals between the predicted and the actual values (pred - true!)
    
    Parameters:
    ----------
    model_name : str
        name of the model that has been used for this prediction/run
    y_true : pd.DataFrame
        the dataframe containing the "true" target variables
    y_pred: pd.DataFrame
        predicted dataframe containing the target
    bins : str
        number of bins the histogram shall have. 30 is the standard.

    Returns:
    -------
    None
        Plot as described
    """
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, bins=bins)
    plt.title(f"{model_name} — Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.show()
    
    return None


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
    # check for task_type first (as defined in evaluate_run())
    if result_dict["task_type"] == "classification":
        metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    elif result_dict["task_type"] == "regression":
        metrics = ["r2", "mse", "mae", "mape"]
    else:
        raise ValueError(f"No valid task_type provided: {result_dict["task_type"]}")
    
    # extract the values to plot
    values = [result_dict[m] for m in metrics]

    # Plotting
    plt.figure(figsize=figsize)

    bars = plt.bar(metrics, values, color='skyblue', edgecolor='black')

    plt.ylim(0, 1)
    plt.title(f"Performance Metrics — {result_dict['name']}")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tick_params(axis="x", labelrotation = x_rotation)

    # Add values to the bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return None