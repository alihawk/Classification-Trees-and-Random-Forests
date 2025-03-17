# Classification Trees and Random Forests for TKI Resistance Data

This project implements classification trees and random forests in Python for analyzing FTIR spectral data related to TKI resistance. The project demonstrates:

- **Classification Trees:** A flexible decision tree algorithm that uses the Gini impurity criterion to split the data. The tree stops splitting when the number of samples in a node is below a threshold (min_samples=2) or when the node is pure.
- **Random Forests:** An ensemble of trees built on bootstrap samples. At each split, a random subset of features (specifically, √(d) features where d is the total number of features) is considered. The random forest reduces overfitting by averaging the predictions of multiple trees.
- **Permutation-based Variable Importance:** Calculation of feature importance by measuring the increase in prediction error when a feature's values are permuted in the out-of-bag (OOB) samples.
- **3-Feature Combination Analysis:** Two methods are implemented:
  - **Data-based:** Ranks features by single-feature importance, then evaluates all combinations of the top features.
  - **Structure-based:** Counts the frequency of feature combinations based on the splits used in the trees.
  
Additionally, the project includes unit tests to verify that the implementation conforms to the required interface and functions correctly on a small dataset.

The code is compatible with Python 3.12 and relies on standard libraries such as NumPy, CSV, and Matplotlib.

## Files Included

- `hw_tree.py`: The main Python code implementing the decision tree, random forest, and variable importance methods.
- `test_hw_tree.py`: Unit tests for verifying the functionality of the implementation.
- `tki-train.tab` and `tki-test.tab`: The TKI resistance FTIR spectral data files used for training and testing.
- `report.pdf` (or the LaTeX source files): The project report explaining the methodology, experiments, and results.

## How to Run

To run the project from the command line, navigate to the project directory and execute:

```bash
python hw_tree.py --num_trees 100 --num_trees_part3 1000
