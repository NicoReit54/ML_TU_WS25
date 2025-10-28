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
│       ├── create_correlation_matrix()
│       └── create_scatterplot_matrix()
│       └── create_countplot()
├── data/
│   └── wine_data/
│       └── wine_data.xlsx
│   └── ...
```

## Importing utils functuons

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
