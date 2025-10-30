import pandas as pd
import uuid

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_run(classifier: str, 
                 y_true: pd.DataFrame, 
                 y_pred: pd.DataFrame,
                 id: str = "") -> dict:
    """
        Computes standard metrics and put into a result dict for further usage.
    
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
        recall=recall_score(y_true, y_pred, average="macro")
    )