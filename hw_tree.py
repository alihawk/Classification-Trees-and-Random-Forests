# Import necessary modules
import csv  # For reading CSV or tab-delimited files
import numpy as np  # For numerical operations (arrays, math functions)
import random  # For random number generation and shuffling
import argparse  # For parsing command-line arguments

# Try to import matplotlib for plotting (only needed in the __main__ block)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------------------------------------------------------
# Helper Functions and Classes
# -----------------------------------------------------------------------------

def gini(y):
    """
    Compute the Gini impurity for an array of labels.
    The Gini impurity is calculated as: 1 - (p^2 + (1 - p)^2),
    where p is the fraction of samples belonging to class 0.

    Example:
      If y = [0, 0, 1, 1], then p = 0.5, and the Gini impurity is 1 - (0.25 + 0.25) = 0.5.
    """
    if len(y) == 0:
        return 0
    p = np.mean(y == 0)  # Calculate proportion of class 0
    return 1 - (p ** 2 + (1 - p) ** 2)


class Node:
    """
    Represents a node in the decision tree.

    If the node is a leaf, 'is_leaf' is True and 'prediction' stores the predicted class.
    Otherwise, 'feature' holds the index of the feature used for splitting,
    'threshold' holds the value that splits the data, and 'left' and 'right' are pointers
    to the child nodes (subtrees).
    """

    def __init__(self, is_leaf, prediction=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf  # Boolean flag indicating if this is a leaf node
        self.prediction = prediction  # Prediction value if it's a leaf node
        self.feature = feature  # Feature index used for splitting (if not a leaf)
        self.threshold = threshold  # Threshold value used to split the feature
        self.left = left  # Left child node (for samples with feature value < threshold)
        self.right = right  # Right child node (for samples with feature value >= threshold)


def all_columns(X, rand):
    """
    Returns an iterator over all column indices of the array X.
    This function is used when we want to consider all features for splitting.

    Example:
      If X has shape (100, 50), this returns range(50).
    """
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    """
    Returns a random subset of feature indices.
    The number of features returned is the square root of the total number of features.
    This is useful for random forests when only a subset of features is considered at each split.

    Example:
      If X has 64 features, int(np.sqrt(64)) equals 8. This function returns 8 randomly selected indices.
    """
    n_features = X.shape[1]
    num = max(1, int(np.sqrt(n_features)))  # Ensure at least one feature is chosen
    indices = list(range(n_features))
    rand.shuffle(indices)  # Shuffle the indices randomly using the provided random generator
    return indices[:num]  # Return the first 'num' indices from the shuffled list


# -----------------------------------------------------------------------------
# Tree and TreeModel Classes
# -----------------------------------------------------------------------------

class Tree:
    def __init__(self, rand=None, get_candidate_columns=all_columns, min_samples=2, max_depth=None):
        """
        Initialize the decision tree.
        - rand: a random number generator (for reproducibility).
        - get_candidate_columns: function to select features for splitting (allows randomness).
        - min_samples: the minimum number of samples to allow a split.
        - max_depth: maximum depth of the tree; if None, the tree grows until stopping criteria are met.
        """
        self.rand = rand if rand is not None else random.Random()
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.max_depth = max_depth

    def build(self, X, y):
        """
        Build the decision tree using the training data X and labels y.
        Returns a TreeModel, which encapsulates the tree and provides a prediction method.
        """
        # Start building from the root with depth 0
        root = self._build_tree(X, y, depth=0)
        return TreeModel(root)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree.
        X: training samples (NumPy array)
        y: corresponding labels
        depth: current depth in the tree

        Returns a Node.
        """
        n_samples = X.shape[0]  # Number of samples at this node

        # Check if we should stop splitting: if too few samples, or the node is pure, or maximum depth is reached.
        if n_samples < self.min_samples or gini(y) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            # Use majority vote as the prediction for this leaf node.
            prediction = int(round(np.mean(y)))
            return Node(is_leaf=True, prediction=prediction)

        # Determine which features (columns) to consider for splitting
        candidate_columns = self.get_candidate_columns(X, self.rand)
        best_feature = None  # Best feature to split on
        best_threshold = None  # Best threshold value for that feature
        best_impurity = float('inf')  # Start with an infinitely bad impurity
        best_splits = None  # Will hold boolean masks for left/right splits
        current_impurity = gini(y)  # Gini impurity of the current node

        # Loop through each candidate feature
        for feature in candidate_columns:
            # Get unique sorted values for the feature
            values = np.sort(np.unique(X[:, feature]))
            if len(values) == 1:
                continue  # If the feature is constant, skip it.
            # Compute potential thresholds as midpoints between adjacent unique values.
            thresholds = (values[:-1] + values[1:]) / 2.0
            # If there are too many candidate thresholds, sample only 10 evenly spaced ones:
            if len(thresholds) > 10:
                idx = np.linspace(0, len(thresholds) - 1, num=10, dtype=int)
                thresholds = thresholds[idx]
            # For each candidate threshold:
            for threshold in thresholds:
                left_idx = (X[:, feature] < threshold)  # Boolean array for left split
                right_idx = ~left_idx  # Complement for right split
                # Skip if one of the splits is empty
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                # Compute impurity for left and right splits
                impurity_left = gini(y[left_idx])
                impurity_right = gini(y[right_idx])
                # Compute weighted impurity of the split:
                weighted_impurity = ((np.sum(left_idx) * impurity_left + np.sum(right_idx) * impurity_right)
                                     / n_samples)
                # If this split is better than previous splits, update the best parameters.
                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (left_idx, right_idx)

        # If no valid split was found or no improvement in impurity is possible:
        if best_feature is None or best_impurity >= current_impurity:
            prediction = int(round(np.mean(y)))
            return Node(is_leaf=True, prediction=prediction)

        # Recursively build the left and right subtrees using the best split found.
        left_node = self._build_tree(X[best_splits[0]], y[best_splits[0]], depth + 1)
        right_node = self._build_tree(X[best_splits[1]], y[best_splits[1]], depth + 1)
        # Return an internal node with the selected splitting feature and threshold.
        return Node(is_leaf=False, feature=best_feature, threshold=best_threshold,
                    left=left_node, right=right_node)


class TreeModel:
    def __init__(self, root):
        """
        Initialize the TreeModel.
        Stores the root of the tree and collects all features used in splits.
        """
        self.root = root
        self.used_features = set()  # This set will contain indices of features used for splitting in the tree.
        self._collect_used_features(root)

    def _collect_used_features(self, node):
        """
        Recursively traverse the tree to collect features used in splits.
        For each non-leaf node, add its 'feature' to self.used_features.
        This is useful later for determining which trees even use a given feature.
        """
        if not node.is_leaf:
            self.used_features.add(node.feature)
            self._collect_used_features(node.left)
            self._collect_used_features(node.right)

    def predict(self, X):
        """
        Predict the class labels for input samples X.
        Loops over all samples and predicts by traversing the tree.
        """
        preds = []
        for i in range(X.shape[0]):
            preds.append(self._predict_sample(self.root, X[i]))
        return np.array(preds)

    def _predict_sample(self, node, x):
        """
        Recursively traverse the tree for a single sample x.
        At each internal node, decide to go left or right based on the feature value and threshold.
        Returns the prediction from the leaf node.
        """
        if node.is_leaf:
            return node.prediction
        if x[node.feature] < node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)


# -----------------------------------------------------------------------------
# Random Forest and RFModel Classes (with in-bag tracking)
# -----------------------------------------------------------------------------

class RandomForest:
    def __init__(self, rand=None, n=100):
        """
        Initialize the random forest.
        - n: number of trees in the forest.
        - rand: random number generator for reproducibility.
        """
        self.rand = rand if rand is not None else random.Random()
        self.n = n
        # Create a base tree that will be used by each bootstrap sample.
        self.rftree = Tree(
            rand=self.rand,
            get_candidate_columns=random_sqrt_columns,  # Use random feature subset at each split
            min_samples=2,
            max_depth=None
        )

    def build(self, X, y):
        """
        Build the random forest:
         - For each tree, a bootstrap sample (with replacement) is drawn from X.
         - The indices used for this tree (in-bag) are recorded.
         - A tree is built on the bootstrap sample.
        Returns an RFModel containing all trees and their in-bag masks.
        """
        n_samples = X.shape[0]
        trees = []  # List to store each tree's model
        inbag_masks = []  # List to store in-bag masks (boolean arrays) for each tree

        # Loop to build each tree
        for i in range(self.n):
            # Create a local random generator for this tree using a random seed
            local_rand = random.Random(self.rand.randint(0, 10 ** 6))
            # Generate bootstrap indices (sample with replacement)
            indices = [local_rand.randrange(n_samples) for _ in range(n_samples)]
            # Create an in-bag mask: True if the sample is used in the bootstrap sample
            used_mask = np.zeros(n_samples, dtype=bool)
            used_mask[indices] = True
            # Select the bootstrap sample from X and y
            X_sample = X[indices]
            y_sample = y[indices]
            # Build a tree on this bootstrap sample (using random feature selection)
            tree = Tree(
                rand=local_rand,
                get_candidate_columns=random_sqrt_columns,
                min_samples=2,
                max_depth=None
            )
            tree_model = tree.build(X_sample, y_sample)
            # Append the built tree and its in-bag mask to the lists
            trees.append(tree_model)
            inbag_masks.append(used_mask)
        # Return an RFModel which encapsulates all the trees and the original training data
        return RFModel(trees, X, y, self.rand, inbag_masks)


class RFModel:
    def __init__(self, trees, X, y, rand, inbag_masks):
        """
        Initialize the RFModel.
        - trees: list of decision tree models built from bootstrap samples.
        - X, y: the original training data and labels.
        - inbag_masks: list of boolean arrays indicating which samples were used (in-bag) for each tree.
        """
        self.trees = trees
        self.X = X
        self.y = y
        self.rand = rand
        self.inbag_masks = inbag_masks

    def predict(self, X):
        """
        Predict the class label for each sample in X by aggregating votes from all trees.
        For each sample, each tree gives a prediction; the final prediction is the majority vote.
        """
        preds = []
        # Loop over each sample
        for i in range(X.shape[0]):
            # For the i-th sample, get the prediction from each tree
            votes = [tree.predict(X[i:i + 1])[0] for tree in self.trees]
            # Count votes for each class (assumes classes are 0 and 1)
            vote_counts = np.bincount(votes, minlength=2)
            # If there is a tie, default to class 0; otherwise, take the class with more votes.
            if vote_counts[0] == vote_counts[1]:
                pred = 0
            else:
                pred = np.argmax(vote_counts)
            preds.append(pred)
        return np.array(preds)

    def predict_oob(self):
        """
        Predict the class label for each training sample using only the trees where the sample was NOT in the bootstrap sample.
        This is known as out-of-bag (OOB) prediction.
        Steps:
         1. For each tree, determine which samples are OOB (not used for training that tree).
         2. Use those trees to predict the label for each OOB sample.
         3. Aggregate votes across trees to obtain the final OOB prediction.
        """
        n_samples = self.X.shape[0]
        # Create a list for each sample to hold votes from trees where the sample is OOB.
        all_votes = [[] for _ in range(n_samples)]
        # Loop over each tree
        for t_idx, tree in enumerate(self.trees):
            used_mask = self.inbag_masks[t_idx]  # Boolean mask for in-bag samples for tree t_idx
            oob_mask = ~used_mask  # OOB samples: logical NOT of used_mask
            if not np.any(oob_mask):
                continue  # If no sample is OOB for this tree, skip it
            # Get OOB samples for this tree and predict their labels
            X_oob = self.X[oob_mask]
            preds = tree.predict(X_oob)
            # Find the indices of OOB samples in the original dataset
            oob_indices = np.where(oob_mask)[0]
            # For each OOB sample, add its prediction to the list of votes
            for i, idx in enumerate(oob_indices):
                all_votes[idx].append(preds[i])
        # Now, determine final OOB prediction for each sample by majority vote
        final_preds = np.full(n_samples, -1, dtype=int)  # Initialize predictions with -1 for samples with no OOB votes
        for i in range(n_samples):
            if len(all_votes[i]) == 0:
                continue  # If no votes, leave prediction as -1
            vote_counts = np.bincount(all_votes[i], minlength=2)
            if vote_counts[0] == vote_counts[1]:
                final_preds[i] = 0  # Tie-breaking: choose class 0
            else:
                final_preds[i] = np.argmax(vote_counts)
        return final_preds

    def importance(self):
        """
        Compute single-feature permutation-based variable importance using OOB samples.
        For each feature:
         1. Obtain the baseline OOB error using predictions from trees for which the sample is OOB.
         2. For each tree that used a given feature, permute the values of that feature in its OOB samples.
         3. Compute the error after permutation.
         4. The increase in error (permuted error - baseline error) is the importance of that feature.
        """
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        baseline_oob = self.predict_oob()  # Get baseline predictions using OOB samples
        valid_mask = (baseline_oob != -1)  # Identify samples that received any OOB prediction
        if not np.any(valid_mask):
            return np.zeros(n_features)
        # Calculate baseline error: fraction of OOB predictions that do not match true labels.
        baseline_error = np.mean(baseline_oob[valid_mask] != self.y[valid_mask])
        importances = np.zeros(n_features)  # To store importance for each feature
        # Loop over every feature
        for f in range(n_features):
            X_perm = self.X.copy()  # Create a copy of the training data to permute feature f
            # For each tree, if the tree used feature f, permute its OOB samples for that feature.
            for t_idx, tree in enumerate(self.trees):
                if f not in tree.used_features:
                    continue  # Skip trees that did not use feature f at all
                used_mask = self.inbag_masks[t_idx]
                oob_mask = ~used_mask
                if np.sum(oob_mask) > 0:
                    idx_oob = np.where(oob_mask)[0]
                    # Permute the indices within the OOB samples for feature f
                    perm = np.random.permutation(len(idx_oob))
                    X_perm[idx_oob, f] = X_perm[idx_oob[perm], f]
            # Now, get predictions on the training data with the permuted feature f
            perm_preds = []
            for i in range(n_samples):
                votes = []
                for t_idx, tree in enumerate(self.trees):
                    # Only consider trees where sample i was OOB
                    if not self.inbag_masks[t_idx][i]:
                        votes.append(tree.predict(X_perm[i:i + 1])[0])
                if len(votes) == 0:
                    perm_preds.append(-1)
                else:
                    vote_counts = np.bincount(votes, minlength=2)
                    if vote_counts[0] == vote_counts[1]:
                        perm_preds.append(0)
                    else:
                        perm_preds.append(np.argmax(vote_counts))
            perm_preds = np.array(perm_preds)
            valid_perm = (perm_preds != -1) & valid_mask
            if np.any(valid_perm):
                # Compute error after permutation
                perm_error = np.mean(perm_preds[valid_perm] != self.y[valid_perm])
                # The importance is the increase in error compared to the baseline
                importances[f] = perm_error - baseline_error
        return importances

    def importance3(self, top_k=5, n_subset=50):
        """
        Compute permutation-based importance for combinations of 3 features.
        This version uses full data predictions (not OOB).
        Steps:
         1. Compute single-feature importance and select the top_k features.
         2. Randomly select a subset of samples (of size n_subset).
         3. For each combination (triple) of features among these top features:
            - Permute the values for these features in the subset.
            - Compute the prediction error on the subset.
         4. The importance of the triple is the increase in error compared to the baseline error.
         5. Return a dictionary mapping each triple to its importance and also the best triple.
        """
        from itertools import combinations
        single_imp = self.importance()  # Get single-feature importances
        # Get indices of the top_k most important features (in descending order)
        top_features = np.argsort(-single_imp)[:top_k]
        n = self.X.shape[0]
        sample_size = min(n_subset, n)
        # Randomly select a subset of sample indices
        subset_idx = np.random.choice(n, sample_size, replace=False)
        X_subset = self.X[subset_idx]
        y_subset = self.y[subset_idx]
        baseline_pred = self.predict(X_subset)  # Baseline predictions on the subset
        baseline_error = np.mean(baseline_pred != y_subset)
        combo_imp = {}  # To store importance of each triple
        best_combo = None
        best_val = -np.inf
        # Generate all possible combinations of 3 features among the top_features
        for combo in combinations(top_features, 3):
            X_perm = X_subset.copy()
            # Permute each feature in the triple independently
            for f in combo:
                perm = np.random.permutation(sample_size)
                X_perm[:, f] = X_perm[perm, f]
            pred = self.predict(X_perm)
            err = np.mean(pred != y_subset)
            imp = err - baseline_error  # Increase in error due to permutation
            combo_imp[combo] = imp
            if imp > best_val:
                best_val = imp
                best_combo = combo
        return combo_imp, best_combo

    def importance3_structure(self):
        """
        Structure-based approach to determine the best triple of features.
        For each tree, we collect the set of features used in its splits.
        Then, we count how many times each 3-feature combination occurs across all trees.
        The triple with the highest count is returned.
        This method does not rely on error measurements but on the tree structure.
        """
        from itertools import combinations
        combo_counts = {}
        # Loop over each tree
        for tree in self.trees:
            feats = tree.used_features  # Get features used in this tree
            # Generate all sorted 3-feature combinations from the used features
            for combo in combinations(sorted(feats), 3):
                combo_counts[combo] = combo_counts.get(combo, 0) + 1
        if not combo_counts:
            return None
        # Select the triple with the maximum frequency (most common combination)
        best_combo = max(combo_counts, key=combo_counts.get)
        return best_combo

    def plot_importance(self, legend, top_n=20):
        """
        Plot the top_n features by their computed importance.
        The feature names are taken from the provided legend list.
        """
        if plt is None:
            print("matplotlib is not available.")
            return
        imp = self.importance()
        sorted_idx = np.argsort(-np.abs(imp))
        top_n = min(top_n, len(sorted_idx))
        top_idx = sorted_idx[:top_n]
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(top_idx))
        labels = [legend[i] for i in top_idx]
        plt.barh(y_pos, imp[top_idx], color='skyblue')
        plt.yticks(y_pos, labels)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Variable Importance (OOB)")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Evaluation Functions
# -----------------------------------------------------------------------------

def hw_tree_full(train, test):
    """
    Build a single decision tree (using min_samples=2) on the training set,
    predict on both training and testing sets, and compute misclassification rates
    along with standard errors.
    """
    X_train, y_train = train
    X_test, y_test = test
    # Initialize the tree with a fixed random seed for reproducibility
    tree = Tree(rand=random.Random(0), get_candidate_columns=all_columns, min_samples=2)
    # Build the tree model using the training data
    model = tree.build(X_train, y_train)
    # Get predictions on training and testing data
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    # Compute misclassification error (fraction of incorrect predictions)
    train_error = np.mean(pred_train != y_train)
    test_error = np.mean(pred_test != y_test)
    # Compute standard errors for the errors (using binomial variance formula)
    train_se = np.sqrt(train_error * (1 - train_error) / len(y_train))
    test_se = np.sqrt(test_error * (1 - test_error) / len(y_test))
    return (train_error, train_se), (test_error, test_se)


def hw_randomforests(train, test, num_trees):
    """
    Build a random forest with 'num_trees' trees, predict on the training and testing sets,
    and compute misclassification rates and standard errors.
    """
    X_train, y_train = train
    X_test, y_test = test
    rf = RandomForest(rand=random.Random(0), n=num_trees)
    # Build the random forest model using training data
    model = rf.build(X_train, y_train)
    # Get predictions on both training and testing data
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    # Compute misclassification error
    train_error = np.mean(pred_train != y_train)
    test_error = np.mean(pred_test != y_test)
    # Compute standard errors
    train_se = np.sqrt(train_error * (1 - train_error) / len(y_train))
    test_se = np.sqrt(test_error * (1 - test_error) / len(y_test))
    return (train_error, train_se), (test_error, test_se)


def hw_rf_importance(train, legend=None):
    """
    Build a random forest with 100 trees, compute variable importance using OOB samples,
    and optionally plot the importances using the provided legend.
    """
    X_train, y_train = train
    rf = RandomForest(rand=random.Random(0), n=100)
    model = rf.build(X_train, y_train)
    imp = model.importance()
    if legend is not None and plt is not None:
        model.plot_importance(legend=legend)
    return imp


def hw_nonrandom_importance(train):
    """
    Build 100 trees using all features (non-random splitting) on bootstrap samples and
    compute variable importance using OOB samples.
    """
    X_train, y_train = train
    n_samples = X_train.shape[0]
    trees = []
    inbag_masks = []
    global_rand = random.Random(0)
    for i in range(100):
        local_rand = random.Random(global_rand.randint(0, 10 ** 6))
        # Generate bootstrap sample indices
        indices = [local_rand.randrange(n_samples) for _ in range(n_samples)]
        used_mask = np.zeros(n_samples, dtype=bool)
        used_mask[indices] = True
        X_sample = X_train[indices]
        y_sample = y_train[indices]
        # Build tree using all features (using all_columns function)
        tree = Tree(rand=local_rand, get_candidate_columns=all_columns, min_samples=2, max_depth=None)
        tree_model = tree.build(X_sample, y_sample)
        trees.append(tree_model)
        inbag_masks.append(used_mask)
    model = RFModel(trees, X_train, y_train, global_rand, inbag_masks)
    return model.importance()


def hw_best3_comparison(train, test, num_trees_part3):
    """
    Build a random forest with 'num_trees_part3' trees to select the best 3-feature combinations.
    Compute best triple using both data-based and structure-based methods, then build a single decision
    tree on these three features and compute performance.
    Returns a dictionary with results for each method.
    """
    X_train, y_train = train
    X_test, y_test = test
    rf_large = RandomForest(rand=random.Random(0), n=num_trees_part3)
    forest = rf_large.build(X_train, y_train)
    # Get best triple based on permutation-based importance (data-based)
    _, best_combo_data = forest.importance3(top_k=5, n_subset=50)
    # Get best triple based on structure-based approach
    best_combo_struct = forest.importance3_structure()
    results = {}
    # Evaluate performance for each method’s best triple
    for label, combo in [('Data-based', best_combo_data), ('Structure-based', best_combo_struct)]:
        if combo is None:
            results[label] = (None, None)
        else:
            # Select only the columns corresponding to the best triple
            X_train_3 = X_train[:, list(combo)]
            X_test_3 = X_test[:, list(combo)]
            tree = Tree(rand=random.Random(0), get_candidate_columns=all_columns, min_samples=2)
            model = tree.build(X_train_3, y_train)
            pred_train = model.predict(X_train_3)
            pred_test = model.predict(X_test_3)
            train_error = np.mean(pred_train != y_train)
            test_error = np.mean(pred_test != y_test)
            train_se = np.sqrt(train_error * (1 - train_error) / len(y_train))
            test_se = np.sqrt(test_error * (1 - test_error) / len(y_test))
            results[label] = ((train_error, train_se), (test_error, test_se), combo)
    return results


# -----------------------------------------------------------------------------
# Data Reading Functions
# -----------------------------------------------------------------------------

def read_tab(fn, adict):
    """
    Read a tab-separated file.
    The first row is assumed to be a header; the first column is a label.
    The remaining columns are converted to a NumPy array of floats.

    Parameters:
      fn: filename (string)
      adict: a dictionary mapping the first column values to numeric labels

    Returns:
      legend: list of feature names (header, excluding the label column)
      X: NumPy array of data (features)
      y: NumPy array of labels
    """
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))
    legend = content[0][1:]  # Skip the first column which contains labels
    data = content[1:]
    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])
    return legend, X, y


def tki():
    """
    Read the TKI training and testing data files.

    Returns:
      (X_train, y_train): training data and labels,
      (X_test, y_test): testing data and labels,
      legend: list of feature names.
    """
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="HW Tree: Classification Trees & Random Forests with Part 3 Extensions")
    parser.add_argument("--num_trees", type=int, default=100,
                        help="Number of trees for the standard random forest (Parts 1 and 2)")
    parser.add_argument("--num_trees_part3", type=int, default=1000,
                        help="Number of trees for the Part 3 forest")
    args, unknown = parser.parse_known_args()

    if plt is not None:
        plt.ion()  # Enable interactive plotting mode

    # Load data from TKI files
    learn, test, legend = tki()

    # -------------------------
    # Part 1: Evaluate Models
    # -------------------------
    # Evaluate a single decision tree
    tree_results = hw_tree_full(learn, test)
    # Evaluate a random forest
    rf_results = hw_randomforests(learn, test, args.num_trees)

    print("=== Full Decision Tree ===")
    print(f"Training Error: {tree_results[0][0] * 100:.2f}% (SE: {tree_results[0][1] * 100:.2f}%)")
    print(f"Test Error: {tree_results[1][0] * 100:.2f}% (SE: {tree_results[1][1] * 100:.2f}%)")
    print(f"\n=== Random Forest ({args.num_trees} trees) ===")
    print(f"Training Error: {rf_results[0][0] * 100:.2f}% (SE: {rf_results[0][1] * 100:.2f}%)")
    print(f"Test Error: {rf_results[1][0] * 100:.2f}% (SE: {rf_results[1][1] * 100:.2f}%)")

    # -------------------------
    # Part 2: Variable Importance
    # -------------------------
    rf_importance = hw_rf_importance(learn, legend=legend)
    nonrandom_importance = hw_nonrandom_importance(learn)

    if plt is not None:
        # Create two subplots for comparing importance from RF and non-random trees
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for Random Forest Importance:
        sorted_idx_rf = np.argsort(-np.abs(rf_importance))
        top_n = min(20, len(sorted_idx_rf))
        top_idx_rf = sorted_idx_rf[:top_n]
        y_pos_rf = np.arange(len(top_idx_rf))
        feature_names_rf = [legend[i] for i in top_idx_rf]
        axes[0].barh(y_pos_rf, rf_importance[top_idx_rf], color='skyblue')
        axes[0].set_yticks(y_pos_rf)
        axes[0].set_yticklabels(feature_names_rf)
        axes[0].set_xlabel("Importance")
        axes[0].set_ylabel("Feature")
        axes[0].set_title(f"Random Forest ({args.num_trees} trees) OOB Importance")

        # Plot for Non-Random Tree Importance:
        sorted_idx_nr = np.argsort(-np.abs(nonrandom_importance))
        top_idx_nr = sorted_idx_nr[:top_n]
        y_pos_nr = np.arange(len(top_idx_nr))
        feature_names_nr = [legend[i] for i in top_idx_nr]
        axes[1].barh(y_pos_nr, nonrandom_importance[top_idx_nr], color='orange')
        axes[1].set_yticks(y_pos_nr)
        axes[1].set_yticklabels(feature_names_nr)
        axes[1].set_xlabel("Importance")
        axes[1].set_ylabel("Feature")
        axes[1].set_title("Non-Random Tree (100 trees) OOB Importance")

        plt.tight_layout()
        # Save the figure as an SVG file
        plt.savefig("part2_importance.svg", format="svg")
        # Optionally, if using Google Colab, download the file automatically
        try:
            from google.colab import files

            files.download("part2_importance.svg")
        except Exception as e:
            pass
        plt.show()

    # -------------------------
    # Part 3: Best 3-Variable Comparison
    # -------------------------
    best3_results = hw_best3_comparison(learn, test, args.num_trees_part3)
    if best3_results:
        print("\n=== Best 3-Variable Comparison ===")
        for method, res in best3_results.items():
            if res[0] is None:
                print(f"{method}: No triple selected.")
            else:
                ((tr_err, tr_se), (te_err, te_se), combo) = res
                print(f"{method} Best Triple: {combo}")
                print(f"  Training Error: {tr_err * 100:.2f}% (SE: {tr_se * 100:.2f}%)")
                print(f"  Test Error: {te_err * 100:.2f}% (SE: {te_se * 100:.2f}%)")
        if plt is not None:
            methods = list(best3_results.keys())
            test_errors = [best3_results[m][1][0] * 100 for m in methods if best3_results[m][0] is not None]
            plt.figure(figsize=(6, 4))
            plt.bar(methods, test_errors, color=['blue', 'orange'])
            plt.ylabel("Test Error (%)")
            plt.title("Test Error for Best 3-Variable Trees")
            plt.tight_layout()
            plt.savefig("part3_best3_comparison.svg", format="svg")
            try:
                from google.colab import files

                files.download("part3_best3_comparison.svg")
            except Exception as e:
                pass
            plt.show()
    else:
        print("No best 3-variable comparison results available.")
