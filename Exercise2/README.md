# Custom Regression Trees & Random Forests – Exercise 2

## Overview
This exercise is part of our MSc Data Science coursework, where the goal was to implement regression trees and random forests entirely from scratch.  
Each team member coded their own versions independently, which allowed us to compare design decisions, strategies, and performance outcomes.  

For the final submission, we agreed to use **Gregor’s implementation (`src_gregor`)**, while still keeping the other implementation in this repo.

Reason for this choice was the overlapping features of e.g. Gregors implementation with Nicos (in the sense of how the variance was determined) and the overlap in the general strucutre with Lenards. All in all, all implementations work as can be seen in a quick comparison of them in `master_comparison.ipynb`.

## Project Structure
```text
exercise_2/
├── data/
│   ├── steel_industry_data.csv
│   ├── corn_data.csv
│   └── wave_energy.csv
│
│   # now come the three individual implementation. src_gregor was decided upon as a team for the final submission!
├── src_gregor/
│   ├── random_forest.py
│   └── regression_tree.py
│
├── src_lenard/
│   ├── randomforest.py
│   └── decisiontree.py
│
├── src_gregor/
│   ├── testing.ipypy
│   └── customRegressionTreeForest.py
│
├── forest/
│   └── RandomForestRegressor.py
│
├── master_comparison.ipynb # comparison notebook between all implementation
│
├── comparison_notebook.ipynb # comparison notebook for final metric comparison of the custom implementation to other established ones
│
│   # preprocessing scripts for basic preprocessing
├── steel_preproc_preproc.py
├── temp_corn_preproc.py
├── wave_energy_farm_preproc.py
│
├── README.md
└── requirements.txt
```

## Datasets
We evaluated our implementations on three diverse regression datasets:

- **Steel Industry Energy Consumption (UCI)**  
  35,040 instances, 9 features - industrial energy consumption forecasting  

- **Corn Farming Data (Kaggle)**  
  422 instances, 22 features - Agricultural dataset for corn yield prediction  

- **Large‑Scale Wave Energy Farm (UCI)**  
  63,600 instances, 149 features - renewable energy optimization  

Each dataset provided was different, especially in its form (wide, long, short) which is why we decided to go for them.


## Approach
- **Independent implementations**:  
  Every team member coded their own regression tree and random forest from scratch.  
  This gave each of us full insight into the whole development process and of course multiple perspectives on algorithm design, efficiency, and debugging.

- **Final submission**:  
  We decided on **Gregor’s implementation** (`src_gregor`) as the teams baseline as mentioned in the beginning.

- **Preprocessing scripts**:  
  Each dataset has its own preprocessing pipeline (`steel_preproc_preproc.py`, `temp_corn_preproc.py`, `wave_energy_farm_preproc.py`) for basic preprocessing steps.

## Getting Started
Install dependencies:
```bash
pip install -r requirements.txt
```

### Small demo

```python
from src_gregor.regression_tree import DecisionTreeRegressor
from src_gregor.random_forest import RandomForestRegressor
import pandas as pd

# Load data
df = pd.read_csv("data/steel_industry_data.csv")
X = df.drop("Usage_kWh", axis=1).values
y = df["Usage_kWh"].values

# Fit the tree with e.g. max-depth of 5
tree = DecisionTreeRegressor(max_depth=5).fit(X, y)
print(tree.predict(X[:5]))

# Fit the forest with e.g. 10 trees and a max-depth of 5
forest = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X, y)
print(forest.predict(X[:5]))

```

