import numpy as np
import pandas as pd
from typing import Optional, Union, Self

# for parallelization!
from joblib import Parallel, delayed

from .regression_tree import DecisionTreeRegressor


class RandomForestRegressor:
    """
    Random Forest wrapper.
    """
    def __init__(
        self,
        n_estimators: int = 100,
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
        self.trees = []
        self.rng = np.random.default_rng(random_state)
    
    def _fit_tree(self, i: int, X: np.array, y: np.array, idxs:np.array) -> DecisionTreeRegressor:
        '''
        Helperfunction to fit the tree
        within the parallelized way further down below
        
        :param i: iteration sequence
        :param X: training data
        :param y: target
        :param idxs: bootstrap samples if wanted
        :return: fitted DecisionTreeRegressor
        :rtype: DecisionTreeRegressor
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

        self.trees = Parallel(n_jobs=-1)(
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

    def predict(self, X) -> pd.Series:
        # ensure numpy format
        if isinstance(X, pd.DataFrame): X = X.values

        # parallelize, n_jobs=-1 for using all cores avaialble
        tree_preds = Parallel(n_jobs=-1)(
            delayed(tree.predict) (X) 
            for tree in self.trees)
        
        # convert from series to array
        #tree_preds = [pred.values for pred in tree_preds]
        
        # aggregate predictions by averaging
        return np.mean(tree_preds, axis=0)