import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict # for set_params

# for testing and comparison
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for parallelization!
from joblib import Parallel, delayed

# for, well typing
from typing import Literal, Dict, Any, Self
from numpy.typing import ArrayLike
import logging

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

def dict_depth(dic, level = 1):
    
    if not isinstance(dic, dict) or not dic:
        return level
    
    return max(dict_depth(dic[key], level + 1)
                               for key in dic)

from sklearn.base import BaseEstimator, RegressorMixin

class RegressionTreeNico(BaseEstimator, RegressorMixin):
    """
    Custom implementation of a regression decision tree for 
    184.702 Machine Learning (VU 3,0) 2025W

    This class builds a regression tree from scratch and supports both
    categorical and continuous features. It uses variance/MSE reduction
    as the splitting criterion. It also supports recursive tree growth with
    configurable stopping criteria (minimum instances per leaf, maximum depth).
            
    Parameters
    -------
    min_instances : int, optional
        Minimum number of samples required to allow further splitting, else mean is taken.
    max_depth : int, optional
        Maximum depth of the tree. If None, the tree grows until min_instances is reached
    features: list, optional
        Allows the user to choose only a subsample of the data for the fit
    alpha: float, optional
        If passed as != 0, this will enable cost-complexity pruning. 
        For more info look into the respective method "_prune_cost_complexity()"
    """    

    def __init__(self, 
                 min_instances: int = 2,
                 max_depth: int = None,
                 max_features: int = None,
                 features_to_choose: ArrayLike | pd.DataFrame = None,
                 alpha: float = 0.0,
                 random_state=None) -> None:
                 
        self.random_state=random_state
        self.min_instances = min_instances
        self.max_depth = max_depth
        self.max_features = max_features
        self.features_to_choose = features_to_choose
        self.alpha = alpha

    def _calc_MSE(self, Y_true: ArrayLike, Y_pred: ArrayLike) -> np.float64:
        '''
        Calculates mean squared error

        Returns:
            MSE of given features
        
        '''
        return np.square(np.subtract(Y_true, Y_pred)).mean()
    
    def _calc_variance_cat_features(self, data: ArrayLike, target_name: str, which_feature_name: str) -> np.float64:
        '''
        Used to calculate the variance of categorical features to decide which feature to use for the next leaf via minimum variance!

        Parameters
        ----------
        data : ArrayLike
            The data set in which we want to calculate the variance for a given feature
        target_name: str
            Name of the target feature
        which_feature_name: str
            Name of the feature for which we want to calculate the variance.

        Returns
        -------
        np.float64
            The (weighted) variance of the feature in the given data
        '''
        # TODO: Do this all with numpy and avoid the pandas overhead!
        
        # prep building blocks for vectorized variance calculation
        grouped_data = data.groupby(which_feature_name)[target_name]
        sizes = grouped_data.size()
        
        # In case we have only one appearance of a value (size N = 1), 
        # then we have to make sure it makes 0 instead of inf because of the division / (N - 1)
        feature_variances = grouped_data.var(ddof=1).fillna(0) 

        weighted_feature_variances = (sizes / len(data)) * feature_variances

        # return the total weighted variance for the feature
        return weighted_feature_variances.sum()
    
        # old non-vectorized part
        feature_values = np.unique(data[which_feature_name])
        feature_variance = 0

        for value in feature_values:
            subset = data[data[which_feature_name] == value].reset_index()

            # In case we have only one appearance of a value, then we have to make sure it makes 0 instead of inf because of the division / (N - 1)
            if len(subset) <= 1:
                subset_var = 0.0
            else:
                # standard in np.var is divided by N instead of N - 1!
                subset_var = (len(subset)/len(data)) * np.var(subset[target_name], ddof=1)

            feature_variance += subset_var
        
        return feature_variance
    
    def _calc_variance_cont_features(self, data: ArrayLike, target_name: str, which_feature_name: str) -> np.float64:
        '''
        Used to calculate the variance of continous features to decide which feature to use for the next leaf via minimum variance!
        
        Parameters
        ----------
        data : ArrayLike
            The data set in which we want to calculate the variance for a given feature
        target_name: str
            Name of the target feature
        which_feature_name: str
            Name of the feature for which we want to calculate the variance.

        Returns
        -------
        np.float64
            The (weighted) variance of the feature in the given data
        '''   

        # initialize weighted mse
        best_weighted_mse = np.inf
        best_threshold = np.inf

        # removed the following block as it is pandas slicing which is less performant than the numpy series afterwards
        # sort the data by the feature
        #data = data.sort_values(by=which_feature_name)
        #X = data[which_feature_name].values
        #y = data[target_name].values
        feature_idx = data.columns.get_loc(which_feature_name)
        target_idx = data.columns.get_loc(target_name)
        
        data = data.to_numpy()
        X = data[:, feature_idx]
        y = data[:, target_idx]

        # sort the data by the feature
        sort_idx = np.argsort(X)
        X = X[sort_idx]
        y = y[sort_idx]

        # now do all the things we did in the loop in one step!
        # summing up enables us to calculate for each i in a loop to have the sums/means already:
        
        # left side 
        cum_sum = np.cumsum(y)
        cum_squared_sum = np.cumsum(y**2)
        counts = np.arange(1, len(y) + 1) 
        
        # e.g. for i = 4
        # + cum_sum = sum up until 4th row
        # + counts = well.. four

        # mse can also be written (after rearranging) as:
        # 1/N * sum_over_i(y_i**2) - mean**2
        left_mean = cum_sum[:-1] / counts[:-1] # -1: everything but the last as this would be the total sum
        left_mse = (cum_squared_sum[:-1] / counts [:-1]) - left_mean**2 
        
        # total values as prep for right side 
        total_sum = cum_sum[-1]
        total_squared_sum = cum_squared_sum[-1]
        total_count = len(y)

        # right side
        right_sum = total_sum - cum_sum[:-1]
        right_sq_sum = total_squared_sum - cum_squared_sum[:-1]
        right_count = total_count - counts[:-1]

        right_mean = right_sum / right_count
        right_mse = (right_sq_sum / right_count) - right_mean**2

        # weighted MSE for all the different splits
        weighted_mse = (counts[:-1] * left_mse + right_count * right_mse) / total_count

        # "consecutive" rows for the threshold (basically shifting the X up and down visually to overlap consecutively)
        threshold = (X[1:] + X[:-1]) / 2

        best_split = np.argmin(weighted_mse)

        # return the best wmse as well as the threshold
        return weighted_mse[best_split], threshold[best_split]

        # find here the "old" for loop for the exact same functionality as above!

        # compute the mse at current node
        current_yhat = np.mean(data[target_name].mean())
        current_mse = self._calc_MSE(data[target_name], current_yhat)
        current_mse = np.round(current_mse, 3)

        # iterate over all rows of the SORTED data
        for i in range(1, len(data)):
            logger.info(data)
            logger.info(which_feature_name)
            logger.info(f"{data.iloc[i][which_feature_name]}, {data.iloc[i-1][which_feature_name]}")
            # compute average of two consecutive rows to have some threshold start 
            split_val = (data.iloc[i][which_feature_name] + data.iloc[i-1][which_feature_name]) / 2

            # split the data into the two sides
            left_branch = data[data[which_feature_name]<=split_val]
            right_branch = data[data[which_feature_name]>split_val]

            # compute the MSE of both sides
            left_yhat = np.mean(left_branch[target_name]) 
            left_mse = self._calc_MSE(left_branch[target_name], left_yhat) 

            right_yhat = np.mean(right_branch[target_name]) 
            right_mse = self._calc_MSE(right_branch[target_name], right_yhat) 

            # compute weighted MSE
            weighted_mse = ((len(left_branch) * left_mse) + (len(right_branch) * right_mse))/len(data)

            # update bestbest_weighted_mse
            if weighted_mse <= best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_threshold = split_val

        return best_weighted_mse, best_threshold

    def _calc_entropy(self, target_column: str) -> np.float64:
        '''
        Calculate the entropy of a target column
       
        Parameters
        ----------
        target_column : ArrayLike
            The target column of which we want to calculate the entropy

        Returns
        -------
        np.float64
            The entropy of the given target column    
        '''

        elements, counts = np.unique(target_column, return_counts = True)

        entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        
        return entropy
    
    def _calc_information_gain(self, data: ArrayLike, target_name:str, which_feature_name: str) -> np.float64:
        '''
        Calculate the information gain based on an attribute (with its weighted entropy!) and the target variable entropy
        
        Parameters
        ----------
        data : ArrayLike
            The data set in which we want to calculate the information gain for a given feature
        target_name: str
            Name of the target feature
        which_feature_name: str
            Name of the feature for which we want to calculate the information gain.

        Returns
        -------
        np.float64
            The infomation gain of the given column    
        '''
        target_entropy = self._calc_entropy(data[target_name])

        vals, counts = np.unique(data[which_feature_name], return_counts=True)

        # Calculate the WEIGHTED entropy
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * self._calc_entropy(data[data[which_feature_name]==vals[i]].dropna()[target_name]) for i in range(len(vals))])
    
        # Calculate the information gain
        information_gain = target_entropy - weighted_entropy
        
        return information_gain
    
    def _sample_random_features(self, features: list, max_features: int) -> list:
        '''
        Used for sampling a subset of the given features.
        Funfact I originally did this in the RandomForest module on tree level. That way we got high correlation
        of the trees and hence even a decrease in performance!! So we need to do this actually at each node
        in each tree!!

        Paramters
        -------
        features: list
            list of features
        max_features: int
            maximum number of features wished to be kept

        Returns
        -------
            a subset (list) of randomly choosen features based on the max_features nr
        
        
        '''
        if not isinstance(features, list):
            raise ValueError(f"The features need to be passed as a list. It is of type {type(features)}")
        
        # reasoning for "new" way of setting seeds. Not globally but "locally" now
        # https://builtin.com/data-science/numpy-random-seed

        rng = np.random.default_rng(self.random_state)

        # replace=False is quite important as default is True, this would "duplicate" features with bad luck
        random_choice = rng.choice(features, size=int(max_features), replace=False)

        return random_choice
    
    def _find_best_split(self, 
                         data: pd.DataFrame, 
                         target_name: str, 
                         max_features: int = None,
                         categorical_features:list = []) -> Dict:
        best_feature = None
        best_threshold = None
        best_variance = np.inf
        is_continuous = False

        if max_features is not None:
            sample_features = np.append(self._sample_random_features(data.drop(target_name, axis=1).columns.to_list(),
                                                                     max_features=max_features), target_name)
            data = data[sample_features]

        for feature_name in data.columns:
            logger.info(f"passed categorical_features: {categorical_features}")
            if feature_name == target_name:
                continue

            elif feature_name in categorical_features:
                logger.info(f"feature {feature_name} entered categorical processing step")
                variance = self._calc_variance_cat_features(data, target_name, feature_name)
                
                # if improvement: overwrite values
                if variance <= best_variance:
                    best_feature = feature_name
                    best_threshold = None
                    best_variance = variance
                    is_continuous = False
            
            else:
                logger.info(f"feature {feature_name} entered continous processing step")

                variance, threshold = self._calc_variance_cont_features(data, target_name, feature_name)
                                
                # if improvement: overwrite values
                if variance <= best_variance:
                    best_feature = feature_name
                    best_threshold = threshold
                    best_variance = variance
                    is_continuous = True
        
        return best_feature, best_threshold, best_variance, is_continuous

    def _Classfier(
            self, 
            data: pd.DataFrame, 
            target_name: str, 
            min_instances: int = 2, 
            categorical_features: list = [],
            max_features: int = None, 
            max_depth: int = None, 
            features: list = None,
            depth: int = 0) -> dict[str, Any]:
        '''
        Recursive tree building algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Subset of the dataset at the current node.
        target_name : str
            Name of the target column
        min_instances : int, optional
            Minimum number of samples required to allow further splitting, else mean is taken.
        categorical_features : list, optional (but actually necessary in case of such)
            List of feature names that represent categorical features
        max_depth : int, optional
            Maximum depth of the tree. If None, the tree grows until min_instances is reached
        features : list, optional
            List of (remaining) features available for splitting
        depth : int, optional
            Depth of the recursion. Used internally

        Returns
        -------
        dict[str, Any] or float
            A nested dictionary representing the tree structure. 
            "Any" will be another such dictionary or a float (= mean target value) if the leaf node is reached

        
        '''
        # Stopping criterion for the recurrsion if we require some minimum value for the size/length of a sub dataset
        if len(data) < min_instances:
            prediction = float(np.mean(data[target_name]))
            return {
                "samples": len(data),
                "prediction": prediction,
                "error": self._calc_MSE(data[target_name], prediction)
            }
    
        # If the dataset has reached the wished for tdepth, return the mean target feature value of the remaining dataset as before
        elif max_depth != None and depth >= max_depth:
            prediction = float(np.mean(data[target_name]))
            return {
                "samples": len(data),
                "prediction": prediction,
                "error": self._calc_MSE(data[target_name], prediction)
            }

        # Now this is the actual "tree growing" part
        else:
            # get the best split
            best_feature, best_threshold, best_variance, is_continuous = self._find_best_split(data, 
                                                                                               target_name, 
                                                                                               max_features,
                                                                                               categorical_features
                                                                                               )

            features = list(data.columns)
            features.remove(target_name)

            # Small fallback in case of no best value found
            if best_feature == None:
                logger.info(f"No best feature to split on found. \nDepth: {depth} \nFeatures left: {features} ")
                return np.mean(data[target_name])

            if is_continuous:
                # build the branching for continous variables
                left_branch = data[data[best_feature] <= best_threshold]
                right_branch = data[data[best_feature] > best_threshold]
                
                ''' Structure for continous feature split:
                {
                    "feature": "X1",
                    "threshold": 400,
                    "left": { subtree(s) for X <= 400 or value in case we reached conditions (ie leaf node) above},
                    "right": { subtree(s) for X > 400 or value in case we reached conditions (ie leaf node) above},
                    "samples": number of samples at this node,
                    "prediction": mean of the target at this point,
                    "error": mse at this node if it was treated as a leaf
                }   
                '''
                
                # if the new split doesn't reduce the size (i.e. one of the branches has to be empty) we have to stop the recurssion
                if len(left_branch) == 0 or len(right_branch) == 0:
                    prediction = float(np.mean(data[target_name]))
                    return {
                        "samples": len(data),
                        "prediction": prediction,
                        "error": self._calc_MSE(data[target_name], prediction)
                    }
                
                if depth > 100:
                    logger.warning(f"Depth {depth}, feature {best_feature}, threshold {best_threshold}, "
                                f"left={len(left_branch)}, right={len(right_branch)}")
                return {
                    "feature": best_feature,
                    "threshold": best_threshold,
                    "left": self._Classfier(data=left_branch, 
                                            target_name=target_name, 
                                            categorical_features=categorical_features,
                                            min_instances=min_instances, 
                                            max_depth=max_depth, 
                                            depth= depth+1),
                    "right": self._Classfier(data=right_branch, 
                                             target_name=target_name, 
                                             categorical_features=categorical_features,
                                             min_instances=min_instances, 
                                             max_depth=max_depth, 
                                             depth= depth+1),
                    "samples": len(data),
                    "prediction": float(np.mean(data[target_name])),
                    "error": self._calc_MSE(data[target_name], np.mean(data[target_name]))
                }
            
            else:
                branches = {}
                for val in np.unique(data[best_feature]):
                    subset = data[data[best_feature] == val]

                    # If subset is the sane as parent, stop recursion as this gets stuck if the one var is always the best split
                    if len(subset) == len(data):
                        prediction = float(np.mean(data[target_name]))
                        return_dict = {
                            "samples": len(data),
                            "prediction": prediction,
                            "error": self._calc_MSE(data[target_name], prediction)
                        }
                        branches[val] = return_dict
                        continue

                    # keep this safeguard info here if we hit some weird recurssion depth issue again
                    if depth > 100:
                        logger.warning(f"Depth {depth}, value {val}, best_feature {best_feature}, lendata {len(data)}")
                    
                    branches[val] = self._Classfier(data=subset, 
                                                    target_name=target_name, 
                                                    categorical_features=categorical_features,
                                                    min_instances=min_instances, 
                                                    max_depth=max_depth, 
                                                    depth= depth+1)
                
                ''' Structure for non-continous feature split:
                {
                    "feature": "categoryXY",
                    "branches": { 
                        "X": subtree(s) for value "X" of feature categoryXY,
                        "Y": subtree(s) for value "Y" of feature categoryXY
                    },
                    "samples": number of samples at this node,
                    "prediction": mean of the target at this point,
                    "error": mse at this node if it was treated as a leaf
                }   
                '''
                
                return {
                    "feature": best_feature,
                    "branches": branches,
                    "samples": len(data),
                    "prediction": float(np.mean(data[target_name])),
                    "error": self._calc_MSE(data[target_name], np.mean(data[target_name]))
                }
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            categorical_features : list = []) -> Self:
            
        '''
        Creates ("fits") the regression tree to the given dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Training dataset containing the features only
        y: pd.Series
            Training dataset containing the target only
        categorical_features : list, optional (but actually necessary in case of such)
            List of feature names that represent categorical features
        Returns
        -------
        Self
            Stores the learned tree in `self.tree_` of the initialized instance of this class.
        '''
        # added to comply with GridSearchCV interface and not mess with the way I used data in the other methods
        data = X.copy()
        data["_target"] = y
        self.target_name = "_target"
        target_name = "_target"

        # to potentially reuse for normalization
        self.dataset_size = len(data)
        #self.target_name = target_name
        self.categorical_features = categorical_features

        # make sure we do not loose the target (and differentiate here as we also pass np.ndarray due to the random sampling)
        if self.features_to_choose is not None and target_name not in list(self.features_to_choose):
            if isinstance(self.features_to_choose, pd.DataFrame):
                self.features_to_choose.append(target_name)
            elif isinstance(self.features_to_choose, ArrayLike):
                #logger.error(f"{features_to_choose}, {type(features_to_choose)}, {target_name}")
                self.features_to_choose = np.append(self.features_to_choose, target_name)

        # Subsample if wanted
        if self.features_to_choose is not None:
            #logger.error(f"{features_to_choose}, {type(features_to_choose)}")
            data = data[self.features_to_choose]

        self.tree_ = self._Classfier(
            data, target_name, categorical_features=self.categorical_features, 
            min_instances=self.min_instances, max_depth=self.max_depth, max_features=self.max_features
        )
        
        # start pruning if alpha > 0.0 is passed
        if self.alpha > 0.0:
            self.tree_ = self._prune_cost_complexity(self.tree_, self.alpha)
        return self
        
    def _predict_row(self, X_pred: ArrayLike, tree:dict, features:list, categorical_features: list = []):
        logger.info(f"Entering _predict_row with (sub)tree: {tree}")
        logger.info(f"X_pred row: {X_pred}")
        logger.info(f"categorical_features passed to _predict_row: {categorical_features}")
        
        if tree["feature"] in features:
            feature = tree["feature"]
            logger.info(f"Checking for feature: {feature}\ncategorical features passed: {categorical_features}")

            if feature in categorical_features:
                result = tree["branches"][X_pred[feature]]
            else:
                threshold = tree["threshold"]
                logger.info(f"feature: {feature} // feature_value: {X_pred[feature]} // threshold: {threshold}")
                result = tree["left"] if X_pred[feature] <= threshold else tree["right"]
            
            logger.info(f"result: {result}")
            
            # continue if not a leaf
            if isinstance(result, dict):
                return self._predict_row(X_pred, result, features, categorical_features) #.drop(feature)
            # if we get a value then return and "break" the recurssion
            else:
                if isinstance(result, (np.float64, float)):
                    return result
                else:
                    logging.error(f"result is not of type float, type: {type(result)}")
                    raise ValueError(f"result is not of type float, type: {type(result)}")

    def _predict_vectorized(self, X_pred: pd.DataFrame, tree:dict) -> np.ndarray:
        logger.info(f"Entering _predict_vectorized with (sub)tree: {tree}")
       
        # return the leaf of the subset once reached!
        if "branches" not in tree and "threshold" not in tree:
            return np.full(len(X_pred), float(tree["prediction"]))
        
        feature = tree["feature"]

        if "branches" in tree:
            # initialize empty result np array
            y_pred = np.empty(len(X_pred))
            for val, branch in tree["branches"].items():
                mask = X_pred[feature] == val

                # if the mask only contains False, do not go into an infinite loop!
                if not mask.any():
                    continue

                y_pred[mask] = self._predict_vectorized(X_pred[mask], branch)
            return y_pred
        
        elif "threshold" in tree:
            # initialize empty result np array
            y_pred = np.empty(len(X_pred))
            threshold = tree["threshold"]
            mask = X_pred[feature] <= threshold

            # left branch (same condition as before, only go in if we have a non-empty mask)
            if mask.any():
                y_pred[mask] = self._predict_vectorized(X_pred[mask], tree["left"])
            # right branch  
            if (~mask).any():
                y_pred[~mask] = self._predict_vectorized(X_pred[~mask], tree["right"])
            return y_pred
            
        #except Exception as e:
        #    if (len(categorical_features) == 0) and ("branches" in tree): 
        #        logger.error("Categorical feature detected and no categorical_features list passed!")
        #        raise e
        #    raise e
        
    def predict(self, X_pred: pd.DataFrame) -> pd.Series:
        '''
        Vectorized prediction using the fitted regression tree.

        Parameters
        -------
        X_pred : pd.DataFrame
            Input samples with the same columns used during training
        
        Returns
        -------

        '''
        if not hasattr(self, "tree_"):
            raise ValueError("The tree has not been fitted yet, use .fit() method of this class instance first!")
        
        if not isinstance(X_pred, pd.DataFrame): # TODO: Ammend this and make suitable for more types
            raise ValueError(f"As of now, X_pred needs to be of type pd.DataFrame.\nIt is of type {type(X_pred)}")
        
        y_pred = self._predict_vectorized(X_pred, self.tree_)
        # to flatten the np.array and to get an output like with predict_old 
        return pd.Series(y_pred)
        
    def predict_old(self, X_pred: ArrayLike, categorical_features: list = []) -> pd.Series:
    
        '''
        "old" or rather initial implementation with row-wise .apply() execution.
        Quite a bit slower than vectorized operations (ofc)

        Parameters
        -------
        X_pred : pd.DataFrame
            Input samples with the same columns used during training
        categorical_features : list, optional (but actually necessary in case of such)
            List of feature names that represent categorical features
        
        Returns
        -------


        '''

        features = X_pred.columns
        result = X_pred.apply(self._predict_row, args=(self.tree_, features, categorical_features), axis=1)
        return result
    
    def _compute_full_subtree_err(self, tree:dict) -> float:
        '''
        Subtree Error is only the error of the leaves, not the error of every node ofc! (hence commented out below to remind me of that mistake)
        '''
        if "threshold" in tree:
            return self._compute_full_subtree_err(tree["left"]) + self._compute_full_subtree_err(tree["right"]) # + tree["error"]
        elif "branches" in tree:
            #return sum(self._compute_full_subtree_err(branch) + branch["error"] for branch in tree["branches"].values())
            return sum(self._compute_full_subtree_err(branch) for branch in tree["branches"].values())
        else:
            # should be weighted as everything else as tree[error] is based on my 
            # per-sample MSE. Hence multiply by leaf size and divide by total N
            return tree["error"] * tree["samples"] / self.dataset_size
        '''
        {
        "feature": "X1",
        "threshold": 400,
        "left": { subtree(s) for X <= 400 or value in case we reached conditions (ie leaf node) above},
        "right": { subtree(s) for X > 400 or value in case we reached conditions (ie leaf node) above},
        "samples": number of samples at this node,
        "prediction": mean of the target at this point,
        "error": mse at this node if it was treated as a leaf
        }   
        '''      
    
    def _sum_leaves(self, tree: dict) -> int:
        """
        Count leaves in a subtree
        """
        if "threshold" in tree:
            return self._sum_leaves(tree["left"]) + self._sum_leaves(tree["right"])
        elif "branches" in tree:
            return sum(self._sum_leaves(branch) for branch in tree["branches"].values())
        else:
            return 1

    def _prune_cost_complexity(self, tree:dict, alpha:float):
        """
        Goal:
        The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.

        Pruning with the cost-complexity methodology also used in sklearn
        Minimizing: R_alpha(T) = R(T) + alpha * abs(T)
        with ...
        + abs(T) ... Number of leaves
        + R(T) ... Weighted sum of training error of the trees leafes/terminal nodes
        + alpha ... penalty variable to control the strength of the pruning

        We basically at each internal node calculate the "effective" alpha (i.e. setting the above formula equal for single nodes 
        and the branch. The rearanging it for alpha) and prune if it is lower than the choosen penalty alpha.
        This minimizes the the total R_alpha(T)!

        For some reason the pruning only starts kicking in much later than in sklearn, but still works...

        See also:
        https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning

        """

        
        # continous features
        if "threshold" in tree:
            #logging.error(tree["left"])
            tree["left"] = self._prune_cost_complexity(tree["left"], alpha)
            tree["right"] = self._prune_cost_complexity(tree["right"], alpha)

        # categorical features
        elif "branches" in tree:
            for val, branch in tree["branches"].items():
                #logging.error(branch)
                tree["branches"][val] = self._prune_cost_complexity(branch, alpha)
        else:
            return tree

        # Get the stats with our helper functions
        subtree_err = self._compute_full_subtree_err(tree)
        sum_leaves = self._sum_leaves(tree) 
        # should be weighted as everything else as tree[error] is based on my 
        # per-sample MSE. Hence multiply by leaf size and divide by total N
        leaf_err = tree["error"] * tree["samples"] / self.dataset_size # = error in case this node was a leaf

        # effective alpha
        eff_alpha = (leaf_err - subtree_err) / (sum_leaves - 1) if sum_leaves != 1 else np.inf # for case of leaf to be safe
        #eff_alpha /= self.dataset_size
        
        # print(eff_alpha, sum_leaves, subtree_err)
        # prune it if the eff. alpha is lower!
        if eff_alpha <= alpha:
            return {
                "samples": tree["samples"],
                "prediction": tree["prediction"],
                "error": tree["error"]
            }
        
        return tree
    '''
    def get_params(self, deep=True) -> dict[str, Any]:
        return {
            "min_instances": self.min_instances,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "features_to_choose": self.features_to_choose,
            "categorical_features": self.categorical_features,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
    
    def set_params(self, **params) -> Self:
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            setattr(self, key, value)

        return self
    '''
    def print_tree(self, tree: dict = None, depth: int = 0) -> None:
        """
        Recursively prints the regression tree structure
        """
        indent = "  " * depth
        # initialize in first step
        if tree is None:
            tree = self.tree_

        # Continuous split
        if "threshold" in tree:
            print(f"{indent}Feature: {tree['feature']} <= {tree['threshold']}")
            if isinstance(tree["left"], dict):
                print(f"{indent}--> Left:")
                self.print_tree(tree["left"], depth + 1)
            else:
                print(f"{indent}--> Left leaf: {tree['left']:.3f}")

            print(f"{indent}Feature: {tree['feature']} > {tree['threshold']}")
            if isinstance(tree["right"], dict):
                print(f"{indent}--> Right:")
                self.print_tree(tree["right"], depth + 1)
            else:
                print(f"{indent}--> Right leaf: {tree['right']:.3f}")

        # Categorical split
        elif "branches" in tree:
            print(f"{indent}Feature: {tree['feature']} (categorical)")
            for val, branch in tree["branches"].items():
                if isinstance(branch, dict):
                    print(f"{indent}--> Value {val}:")
                    self.print_tree(branch, depth + 1)
                else:
                    print(f"{indent}--> Value {val} leaf: {branch:.3f}")

class RandomForestNico():
    """
    Custom implementation of a regression decision forest for 
    184.702 Machine Learning (VU 3,0) 2025W

    This class builds on the above defined custom regression tree and is basically averaging 
    the predictions of each single tree after applying two basic principles for the training data:
    
    The Random Forest approach is based on two concepts, called *bagging* and *subspace sampling*. 
    Bagging is the short form for *bootstrap aggregation*. 
    Here we create a multitude of datasets of the same length as the 
    original dataset drawn from the original dataset with replacement (the *bootstrap* in bagging).

    For the subspace sampling we only pass a third of the features into each tree. If we pass this 
    max_features int to the tree class we automatically trigger a np.random shuffle/draw of these 
    features. By also passing an incremented integer for the random-state we ensure a different draw 
    each time
     
    We then train a tree model for each of the bootstrapped datasets and take the majority prediction 
    of these models for a unseen query instance as our prediction (the *aggregation* in bagging). 
    
    Here we take the mean or the median for regression tree models and the mode for classification tree models.
    
    ***Source***
    https://python-course.eu/machine-learning/random-forests-in-python.php
    As well as the scipy documentation

    Parameters
    -------
    min_instances : int, optional
        Minimum number of samples required to allow further splitting, else mean is taken.
    max_depth : int, optional
        Maximum depth of the tree. If None, the tree grows until min_instances is reached
    categorical_features : list, optional (but actually necessary in case of such)
        List of feature names that represent categorical features
    nr_of_trees: int
        Define how many trees shall be fit for the RandomForest


    """    
    def __init__(self, 
                 random_state=None,
                 min_instances: int = 2,
                 max_depth: int = None,
                 nr_of_trees: int = 10) -> None:
        self.random_state = random_state
        self.min_instances = min_instances
        self.max_depth = max_depth           
        self.nr_of_trees = nr_of_trees

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            categorical_features: list = []) -> Self:
        '''
        Fit method for training the randomforest model, based on the custom Regressiontree implementation
        
        Parameters
        -------
        X : pd.DataFrame
            Training dataset containing the features only
        y: pd.Series
            Training dataset containing the target only
        categorical_features : list, optional (but actually necessary in case of such)
            List of feature names that represent categorical features
        Returns
        -------
        Self
            Stores the learned forest in `self.forest_` of the initialized instance of this class.
        '''

        # Using the parallelized way taken from https://www.geeksforgeeks.org/python/massively-speed-up-processing-using-joblib-in-python/
        random_forest_trees = Parallel(n_jobs=-1)(
            delayed(RegressionTreeNico(random_state=self.random_state + i,
                                       min_instances = self.min_instances,
                                       max_depth = self.max_depth,
                                       max_features = int(len(X.columns) / 3)).fit) 
            (   
                # Bootstrap Sampling:
                # replace=True ensures sampling WITH replacement (i.e. we have some rows duplicated or similar). frac=1 ensures same data lengths as originally
                X.sample(frac=1, replace=True, random_state=self.random_state + i),
                y.loc[X.sample(frac=1, replace=True, random_state=self.random_state + i).index],
                # By passing max_features we trigger the random sampling mechanism in the tree!
                categorical_features=categorical_features
            )
            for i in range(self.nr_of_trees)
        )

        # list of the trees about to be trained
        '''random_forest_trees = []
        
        for i in range(nr_of_trees):
            # initialize tree class new each time
            tree = RegressionTreeNico(random_state=self.random_state + i)

            # Bootstrap Sampling:
            # replace=True ensures sampling WITH replacement (i.e. we have some rows duplicated or similar). frac=1 ensures same data lengths as originally
            bootstrap_sample = data.sample(frac=1, replace=True, random_state=self.random_state + i)
            #, random_state=self.random_state) >> Dont simply set it like that, because then you'll have the same bootstrapped sample each run

            random_forest_trees.append(
                tree.fit(data = bootstrap_sample, 
                         target_name = target_name, 
                         min_instances = min_instances,
                         max_depth = max_depth,
                         max_features = int(len(bootstrap_sample.columns) / 3),
                         categorical_features=categorical_features)
            )
        '''
        # set as part of the class
        self._random_forest_trees = random_forest_trees

        return self
    
    def predict(self, X_pred: pd.DataFrame) -> pd.Series:
        '''
        
        Parameters
        -------
        X_pred : pd.DataFrame
            Input samples with the same columns used during training
        categorical_features : list, optional (but actually necessary in case of such)
            List of feature names that represent categorical features        

        Returns
        -------
            Averaged predictions based on the predictions of all the trees
        '''

        # Make sure its trained
        if not hasattr(self, "_random_forest_trees"):
            raise ValueError("The random forest has not been trained yet. Use .fit() method of this class instance for that.")
        
        # "Old" non-parallelized way instead of https://www.geeksforgeeks.org/python/massively-speed-up-processing-using-joblib-in-python/
        # Collect all the predictions from the trained trees
        #all_preds = []

        #for tree in self._random_forest_trees:
        #    y_pred = tree.predict(X_pred, categorical_features=categorical_features)
        #    all_preds.append(y_pred.values)

        # n_jobs=-1 for using all cores available
        all_preds = Parallel(n_jobs=-1)(
            delayed(tree.predict) (X_pred) 
            for tree in self._random_forest_trees)

        # convert from series to array
        all_preds = [pred.values for pred in all_preds]

        # Average across all predictions
        mean_preds = np.mean(all_preds, axis=0)
        return pd.Series(mean_preds, index=X_pred.index)