import csv
import numpy as np
import random
import argparse

try:
    import matplotlib.pyplot as pl
except ImportError:
    plt = None


def gini(y):
    if len(y) == 0:
        return 0
    p = np.mean(y==0)
    return 1 - (p**2 + (1-p)**2)

class Node:

    def __init__(self, is_leaf, prediction=None,feature=None, threshold =None, left=None, right=None ):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def all_columns(X,rand):
        return range(X.shape[1])

    def random_sqrt_columns(X, rand):
        n_features = X.shape[1]
        num = max(1,int(np.sqrt(n_features)))
        indices = list(range(n_features))
        random.shuffle(indices)
        return indices[:num]

class Tree:
    def __init__(self,rand=None, get_candidate_columns=all_columns,min_samples=2,max_depth=None):
        self.rand = rand if rand is not None else random.Random()
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.max_depth = max_depth
   def build(self, X, y):
        """
        Build the decision tree and return a TreeModel.
        """
        root = self._build_tree(X, y, depth=0)
        return TreeModel(root)
   def _build_tree(self, X, y , depth):
       n_samples = X.shape[0]
       if n_samples <self.min_samples or gini(y) == 0 or (self.max_depth is not None and depth >= self.max_depth):
           prediction = int(round(np.mean(y)))
           candidate_columns = self.get_candidate_columns(X, self.rand)
           best_feature = None
           best_threshold = None
           best_impurity = float('inf')
           best_splits = None
           current_impurity = gini(y)
           for feature in candidate_columns:
               values = np.sort(np.unique(X[:,feature]))
               if len(values) == 1:
                   continue
                thresholds = (values[:-1] + values[1:])/2.0
               if len(thresholds) > 10:
                   idx = np.linspace(0,len(thresholds) -1 num=10 , dtype =int)








