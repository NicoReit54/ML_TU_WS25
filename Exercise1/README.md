# Notebook & Project Setup Guide

---

## Project Structure

```
project_root/
├── notebooks/
│   ├── wine.ipynb
│   ├── soybean.ipynb
│   └── ...
├── src/
│   └── utils.py
│   │   ├── create_correlation_matrix()
│   │   └── create_scatterplot_matrix()
│   │   └── create_countplot()
│   └── evaluation.py
│       ├── evaluate_run()
│       └── plot_confusion_matrix()
│       └── plot_metrics()
│       └── plot_regression_results()
│       └── plot_residuals()
├── data/
│   └── wine_data/
│       └── wine_data.xlsx
│   └── ...
```

## `utils.py` functions

### Import 

To use functions from `src/utils` inside your notebooks, you have to manually add the project root to Python’s module search path (based on the current project structure)

### Add This to the top of your notebook

```python
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Now you can import your helpers
from src.utils import create_correlation_matrix, create_scatterplot_matrix
```

---

### Using `create_scatterplot_matrix`

```python
def create_scatterplot_matrix(
    df: pd.DataFrame,
    title: str = "Scatterplot Matrix",
    columns: list = None,
    figsize: tuple = (16, 18),
    y_rotation: int = 0,
    y_offset: int = -0.7,
    x_rotation: int = -40) -> None
```

### Parameters

- `df`: DataFrame containing numerical features.
- `columns`: Optional list of columns to include.
- `title`: Plot title.
- `figsize`: Size of the figure.
- `x_rotation`, `y_rotation`: Axis label rotation angles.
- `y_offset`: How much the y labels shall be moved to the left (negative value) or  to the right in order to not overlap with the graph.

### Example Usage

```python
create_scatterplot_matrix(df_wine, figsize=(10, 10), x_rotation=40)
```

### Using `create_correlation_matrix`

```python
def create_correlation_matrix(
    df: pd.DataFrame, 
    column_names: list = None, 
    title: str = "Correlation Matrix",
    columns: list=None,
    figsize: tuple = (10, 10))
```

### Parameters

- `column_names`: Custom labels for axes.
- `df`: DataFrame with numerical columns.
- `title`: Plot title.
- `figsize`: Size of the figure.
- `columns` : Restrict which columns are being used by passing a list of the respective names

### Example Usage

```python
create_correlation_matrix(df.columns.tolist(), df_wine)
```

---
### Using `create_countplot`

```python
def create_countplot(
        df: pd.DataFrame,
        x_rotation: int = 45,
        figsize: tuple = (30, 30)) -> None
```

### Parameters

- `df`: The DataFrame containing features to visualize. Categorical features will be extracted automatically.
- `figsize`: Size of the figure.
- `x_rotation`: Axis label rotation angles.


### Example Usage

```python
create_countplot(df)
```

---

## `evaluation.py` functions

This module provides an evaluation framework for both **classification** and **regression** tasks.  

### Import 

To use functions from `src/evaluation` inside your notebooks, you have to manually add the project root to Python’s module search path (based on the current project structure)

#### Add this to the top of your notebook

```python
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Now you can import your helpers
from src.evaluation import evaluate_run, plot_confusion_matrix, plot_regression_results, plot_residuals, plot_metrics
```
---

### Overview

| Function | Purpose | Task Type |
|-----------|----------|-----------|
| `evaluate_run()` | Computes standard metrics and returns results as a dictionary | classification & regression |
| `plot_confusion_matrix()` | Visualizes confusion matrix from predictions | classification |
| `plot_metrics()` | Displays metric scores as a bar chart | both |
| `plot_regression_results()` | Shows scatterplot of true vs predicted values | regression |
| `plot_residuals()` | Plots the residual (prediction error) distribution | regression |

---

### Example 1 — Regression Task

Example using the **California Housing dataset** and a `LinearRegression` model.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from evaluation import evaluate_run, plot_regression_results, plot_residuals, plot_metrics

# Load dataset and train the model
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate and print the dict
res = evaluate_run(name="LinearRegression", y_true=y_test, y_pred=y_pred, task_type="regression")
print(res)

# Visualize
plot_regression_results(y_true=y_test, y_pred=y_pred, model_name="LinearRegression")
plot_residuals(y_true=y_test, y_pred=y_pred, model_name="LinearRegression")
plot_metrics(res)
```


### Example 2 — Classification Task

Example using the **Iris dataset** and a `RandomForestClassifier`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluate_run, plot_confusion_matrix, plot_metrics

# Load the dataset and train the model
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
res = evaluate_run(name="RandomForest", y_true=y_test, y_pred=y_pred, task_type="classification")
print(res)

# Visualize evaluation results
plot_confusion_matrix(classifier="RandomForest", y_true=y_test, y_pred=y_pred)
plot_metrics(res)
```

### Function Details

#### `evaluate_run()`

```python
evaluate_run(
    name: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    task_type: Literal["classification", "regression"],
    id: str = ""
) -> dict
```

**Description:**
Computes standard metrics and returns them as a dictionary.
+ For regression tasks: R2, MSE, MAE, and MAPE.
+ For classification: accuracy, balanced accuracy, F1, precision, recall, and confusion matrix.

---

#### `plot_confusion_matrix()`

Displays a confusion matrix for classification models.
Automatically labels axes and adds a title with the classifier name.

---

#### `plot_regression_results()`

Plots predicted values vs true targets for regression tasks.
The red dashed line is for indicating "perfect" predictions.

---

#### `plot_residuals()`

Shows the residual/prediction error distribution for regression tasks.

---

#### `plot_metrics()`

Plots a bar chart of the evaluated metrics from `evaluate_run()`
Automatically adjusts depending on task type (classification or regression).

---



