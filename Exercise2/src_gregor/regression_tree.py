import numpy as np
import pandas as pd
from typing import Optional, Union

# for parallelization!
from joblib import Parallel, delayed

# for typing
from typing import Literal, Dict, Any, Self

# used to make the custom class compatible with things like gridsearchcv! 
from sklearn.base import BaseEstimator, RegressorMixin

class Node:
    """
    Represents a node in the decision tree.
    """
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    A Regression Tree that accepts preprocessed (numerical) data.
    Uses optimized variance reduction (O(N) via cumulative sums).
    """
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Self:
        #ensure numpy format
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        #ensure numpy format
        if isinstance(X, pd.DataFrame): X = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        variance = np.var(y) if len(y) > 0 else 0
        
        #stopping Criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or
            variance == 0):
            #create a leaf node
            return Node(value=np.mean(y))

        #feature subsampling (for Random Forest)
        feat_idxs = self._get_feature_indices(n_features)

        #finding best split
        best_split = self._find_best_split(X, y, feat_idxs)

        #if no valid split found, create a leaf node
        if best_split['impurity_gain'] == -1:
            return Node(value=np.mean(y))

        #split data into children and recurse
        left_idxs, right_idxs = best_split['indices']
        left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        #finally, create decision node
        return Node(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['threshold'],
            left=left_child,
            right=right_child
        )

    def _find_best_split(self, X, y, feat_idxs):
        """
        Loop through every selected feature and test to find the best possible split.
        """
        best_split = {'impurity_gain': -1, 'feature_idx': None, 'threshold': None, 'indices': None}
        #total sum of squares
        current_uncertainty = np.var(y) * len(y) 

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            self._find_split(X_col, y, feat_idx, current_uncertainty, best_split)
                
        return best_split

    def _find_split(self, X_col, y, feat_idx, parent_impurity, best_split):
        #sort X and y by the feature value
        sorted_idxs = np.argsort(X_col)
        X_sorted = X_col[sorted_idxs]
        y_sorted = y[sorted_idxs]
        
        #precompute cumulative sums. This allows for O(1) variance calculation later
        n = len(y)
        sum_y = np.cumsum(y_sorted)
        sum_y_sq = np.cumsum(y_sorted ** 2)
        
        total_sum = sum_y[-1]
        total_sum_sq = sum_y_sq[-1]

        #scan for the best split point
        for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
            #skip duplicate values
            if X_sorted[i] == X_sorted[i-1]: 
                continue 

            #left stats
            n_l = i
            sum_l = sum_y[i-1]
            sum_sq_l = sum_y_sq[i-1]
            #calculate SSE for left child
            var_sum_l = sum_sq_l - (sum_l**2 / n_l)

            #right stats
            n_r = n - i
            sum_r = total_sum - sum_l
            sum_sq_r = total_sum_sq - sum_sq_l
            #calculate SSE for right child
            var_sum_r = sum_sq_r - (sum_r**2 / n_r)

            #calculate gain of split
            child_impurity = var_sum_l + var_sum_r
            gain = parent_impurity - child_impurity

            #update best split if gain is better
            if gain > best_split['impurity_gain']:
                best_split.update({
                    'impurity_gain': gain,
                    'feature_idx': feat_idx,
                    'threshold': (X_sorted[i] + X_sorted[i-1]) / 2,
                    'indices': (sorted_idxs[:i], sorted_idxs[i:])
                })

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _get_feature_indices(self, n_total_features):
        """
        Determine which feature indices to consider for the current split.
        """
        #select all features
        if self.max_features is None:
            n_select = n_total_features
        #select int number of features
        elif isinstance(self.max_features, int):
            n_select = self.max_features
        #select float fraction of features
        elif isinstance(self.max_features, float):
            n_select = int(self.max_features * n_total_features)
        #select sqrt number of features
        elif self.max_features == 'sqrt':
            n_select = int(np.sqrt(n_total_features))
        #select log2 number of features
        elif self.max_features == 'log2':
            n_select = int(np.log2(n_total_features))

        else:
            n_select = n_total_features
        
        n_select = max(1, min(n_select, n_total_features))
        #randomly select feature indices
        return self.rng.choice(n_total_features, n_select, replace=False)

