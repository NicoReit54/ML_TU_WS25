import numpy as np
# Import the DecisionTreeRegressor from the decision_tree_regressor.py file
# Assuming it's in the same directory or available in the path
from .decisiontree import DecisionTreeRegressor #needed to add the . for use in master_comparison

class RandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100, n_features=None, bootstrap=True):
        """
        Args:
            n_estimators: Number of trees in the forest
            min_samples_split: Min samples required to split a node (passed to trees)
            max_depth: Max depth of each tree (passed to trees)
            n_features: Number of features to consider when looking for the best split.
                        If None, defaults to all features (or sqrt/log2 logic can be added).
            bootstrap: Whether to use bootstrap samples when building trees.
        """
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X = np.array(X)
        y = np.array(y)
        
        for _ in range(self.n_estimators):
            # 1. Initialize a tree
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            
            # 2. Create Bootstrap Sample (if enabled)
            if self.bootstrap:
                n_samples = X.shape[0]
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y
            
            # 3. Fit the tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        # Collect predictions from every tree
        # Shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Aggregate: Mean of all tree predictions
        # Shape: (n_samples,)
        final_preds = np.mean(tree_preds, axis=0)
        return final_preds
    
