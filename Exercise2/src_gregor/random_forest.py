import numpy as np
import pandas as pd
from typing import Optional, Union, Self

# for parallelization!
from joblib import Parallel, delayed

from .regression_tree import DecisionTreeRegressor

# used to make the custom class compatible with things like gridsearchcv! 
from sklearn.base import BaseEstimator, RegressorMixin

class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Random Forest wrapper.

    This is a simplified implementation of a Random Forest for regression,
    built on top of the custom `DecisionTreeRegressor`. It supports bootstrap
    sampling, feature subsampling, and parallel training of trees.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest
    max_depth : int, default=20
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    max_features : {"sqrt", "log2"}, int, float or None, default=None
        Number of random features to consider when looking for the best split:
        - None: all features
        - int: absolute number of features
        - float: fraction of features
        - "sqrt": square root of total features
        - "log2": log2 of total features
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    trees_ : list of DecisionTreeRegressor
        The collection of all fitted trees in the forest. Underscore to comply with sklearns naming convention
    rng : numpy.random.Generator
        Random number generator instance
    """
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 20,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_ = [] # *_ because this is how we can fit this to the sklearn api with things like gridsearchcv. so it can recognize the item!
        self.rng = np.random.default_rng(random_state)
    
    def _fit_tree(self, i: int, X: np.array, y: np.array, idxs:np.array) -> DecisionTreeRegressor:
        '''
        Fit a single decision tree.

        This helper function is used internally for the parallel training.
        There is potentially/surely a better way but it works for now!

        Parameters
        ----------
        i : int
            Index of the tree in the ensemble (used for seeding)
        X : ndarray of shape (n_samples, n_features)
            Training input samples
        y : ndarray of shape (n_samples,)
            Target values
        idxs : ndarray of shape (n_samples,)
            Indices for bootstrap sampling

        Returns
        -------
        tree : DecisionTreeRegressor
            A fitted decision tree regressor
        '''

        # Bootstrap Sampling:
        if self.bootstrap:
            X_sample, y_sample = X[idxs], y[idxs]
        else:
            X_sample, y_sample = X, y

        # seed per tree different to ensure diversity
        tree_seed = (self.random_state + i) if self.random_state is not None else None

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=tree_seed
        )
        return tree.fit(X_sample, y_sample)

    def fit(self, X, y) -> Self:
        """
        Fits the custom random forest regressor

        Parameters
        ----------
        X : ndarray or DataFrame of shape (n_samples, n_features)
            Training input samples
        y : ndarray or Series of shape (n_samples,)
            Target values

        Returns
        -------
        self : RandomForestRegressor
            Fitted estimator
        """
        # ensure numpy format
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        
        n_samples = X.shape[0]
    
        # precompute bootstrap samples if wanted, could not find a better way but to do it outside of the parallel "loop"
        bootstrap_idxs = [
            self.rng.choice(n_samples, n_samples, replace=True)
            if self.bootstrap else np.arange(n_samples)
            for _ in range(self.n_estimators)
        ]

        self.trees_ = Parallel(n_jobs=-1)(
            delayed(self._fit_tree)(
                i, X, y, bootstrap_idxs[i])
            for i in range(self.n_estimators)
        )

        ''' old non-parallelized way
        self.trees = []
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            #create bootstrap sample, or use full data
            if self.bootstrap:
                idxs = self.rng.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y

            #set different random state for each tree for diversity
            tree_seed = (self.random_state + i) if self.random_state is not None else None
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        '''
        return self

    def predict(self, X) -> np.ndarray:
        # ensure numpy format
        if isinstance(X, pd.DataFrame): X = X.values

        # parallelize, n_jobs=-1 for using all cores avaialble
        tree_preds = Parallel(n_jobs=-1)(
            delayed(tree.predict) (X) 
            for tree in self.trees_)
        
        # convert from series to array
        #tree_preds = [pred.values for pred in tree_preds]
        
        # aggregate predictions by averaging
        return np.mean(tree_preds, axis=0)