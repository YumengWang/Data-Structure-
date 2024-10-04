# EECS 445 - Fall 2024
# Project 1 - project1.py

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)


# Q1a
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: dataframe with columns [Time, Variable, Value]

    Returns:
        a dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'max_HR': 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for _, row in static.iterrows():
        variable = row['Variable']
        value = row['Value']
        feature_dict[variable] = value

    # TODO  3) extract max of time-varying variables into feature dict
    for _, row in timeseries.iterrows():
        variable = row['Variable']
        value = row['Value']
        if variable == 'MechVent': continue
        elif 'max_' + variable not in feature_dict: feature_dict['max_' + variable] = value
        else: feature_dict['max_' + variable] = max(value, feature_dict['max_' + variable])

    for variable in timeseries_variables:
        variable = 'max_' + variable
        if variable not in feature_dict:
            feature_dict[variable] = np.nan
    return feature_dict

# Q1b
def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    # TODO: implement this function according to spec
    for i in range(X.shape[1]):
        column = X[:, i]
        mean = np.nanmean(column)
        column[np.isnan(column)] = mean
        X[:, i] = column
    return X

# Q1c
def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (N, d) matrix.
    
    Returns:
        X: (N, d) matrix. Values are normalized per column.
    """
    # TODO: implement this function according to spec
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function
    and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization (default: None).
        C: Regularization strength parameter (default: 1.0).
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression. 
            Default is 'rbf'.
        gamma (float): Kernel coefficient (default: 0.1).
    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight)
    elif loss == "squared_error":
        return KernelRidge(kernel=kernel, gamma=gamma)


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    metric: str = "accuracy",
    bootstrap: bool=True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X, using 1,000 
    bootstrapped samples of the test set if bootstrap is set to True. Otherwise,
    returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.
    
    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision', 
                'sensitivity', and 'specificity')
    Returns:
        if bootstrap is True: the median performance and the empirical 95% confidence interval in np.float64
        if bootstrap is False: peformance 
    """
    # TODO: Implement this function
    # This is an optional but VERY useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if bootstrap:
        return (0.1, 0.1, 0.1)
    else:
        y_pred = clf_trained.predict(X)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                if y_true[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if y_true[i] == 1:
                    FN += 1
                else:
                    TN += 1
        if metric == "accuracy":
            return (TP + TN) / (TP + TN + FP + FN)
        elif metric == "precision":
            if FP + TP == 0:
                return 0
            else:
                return TP / (FP + TP)
        elif metric == 'f1-score':
            y_score = y_pred * 2 - 1
            return f1_score(y_true, y_score)
        elif metric == 'auroc':
            decision_scores = clf_trained.decision_function(X)
            auroc = roc_auc_score(y_true, decision_scores)
            return auroc
        elif metric == 'average_precision':
            y_score = y_pred * 2 - 1
            return average_precision_score(y_true, y_score)
        elif metric == 'sensitivity':
            if TP + FN == 0:
                return 0
            else:
                return TP / (TP + FN)
        else:
            if TN + FP == 0:
                return 0
            else:
                return TN / (TN + FP)


# Q2.1a
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # TODO: Implement this function

    # NOTE: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # TODO: Return the average, min,and max performance scores across all fold splits in a size 3 tuple.
    skf = StratifiedKFold(n_splits=k)
    performance_list = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        performance_list.append(performance(clf, X_test, y_test, metric, False))

    # TODO: Return the average, min,and max performance scores across all fold splits in a size 3 tuple.
    performance_tuple = (np.mean(performance_list), np.min(performance_list), np.max(performance_list))
    return performance_tuple


# Q2.1b
def select_param_logreg(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over
    
    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # TODO: Implement this function
    # NOTE: You should be using your cv_performance function here
    # to evaluate the performance of each logistic regression classifier

    best_C = 0
    best_penalty = None
    best_performance = float('-inf')
    for C in C_range:
        for penalty in penalties:
            clf = LogisticRegression(penalty=penalty, C=C, solver='liblinear', fit_intercept=False, random_state=seed)
            performances = cv_performance(clf, X, y, metric, k)
            mean_performance = performances[0]
            min_performance = performances[1]
            max_performance = performances[2]
            print(
                f"Performance Measures: C={C}, Penalty={penalty}, Mean={mean_performance:.4f} "f"(Min={min_performance:.4f}, Max={max_performance:.4f}) CV Performance")
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_penalty = penalty
    result = (best_C, best_penalty)
    return result


# Q4c
def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over
    
    Returns:
        The parameter value for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    print(f"RBF Kernel Ridge Regression Model Hyperparameter Selection based on {metric}:")
    # TODO: Implement this function acording to the docstring
    # NOTE: This function should be very similar in structure to your implementation of select_param_logreg()
    return None


# Q2.1e
def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
    
    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        # elements of norm0 should be the number of non-zero coefficients for a given setting of C
        norm0 = []
        for C in C_range:
            # TODO Initialize clf according to C and penalty
            clf = None
            # TODO Fit classifier with X and y

            # TODO: Extract learned coefficients/weights from clf into w
            # Note: Refer to sklearn.linear_model.LogisticRegression documentation
            # for attribute containing coefficients/weights of the clf object
            w = None

            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0
            non_zero_count = None
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    # NOTE: plot will be saved in the current directory
    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    # Read challenge data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()


if __name__ == "__main__":
    main()
