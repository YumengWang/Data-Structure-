# EECS 445 - Fall 2024
# Project 1 - project1.py
import heapq

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


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
        if variable in timeseries_variables:
            key = 'max_' + variable
            if key not in feature_dict:
                feature_dict[key] = value
            else:
                feature_dict[key] = max(value, feature_dict[key])

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


# Q1a
def generate_feature_vector_challenge(df: pd.DataFrame) -> dict[str, float]:
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
        if (variable == 'ICUType'):
            feature_dict['CCU'] = 0
            feature_dict['CSRU'] = 0
            feature_dict['MICU'] = 0
            feature_dict['SICU'] = 0
            if value == 1:
                feature_dict['CCU'] = 1
            elif value == 2:
                feature_dict['CSRU'] = 1
            elif value == 3:
                feature_dict['MICU'] = 1
            elif value == 4:
                feature_dict['SICU'] = 1
        else:
            feature_dict[variable] = value
    # TODO  3) extract max of time-varying variables into feature dict
    features_dict = {}
    for _, row in timeseries.iterrows():
        time = row['Time']
        variable = row['Variable']
        value = row['Value']
        if variable in timeseries_variables:
            if time < '24:00':
                key = 'mean_1_' + variable
                if key not in features_dict:
                    features_dict[key] = [value]
                else:
                    features_dict[key].append(value)
            else:
                key = 'mean_2_' + variable
                if key not in features_dict:
                    features_dict[key] = [value]
                else:
                    features_dict[key].append(value)

    for variable, values in features_dict.items():
        values = [x for x in values if not np.isnan(x)]

        if len(values) == 0:
            feature_dict[variable] = np.nan
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_values = [x for x in values if x >= lower_bound and x <= upper_bound]
        #if (len(filtered_values) == 0):
        #    print("here", values)
        feature_dict[variable] = np.mean(filtered_values)

    for variable in timeseries_variables:
        variable = 'mean_1_' + variable
        if variable not in feature_dict:
            feature_dict[variable] = np.nan

    for variable in timeseries_variables:
        variable = 'mean_2_' + variable
        if variable not in feature_dict:
            feature_dict[variable] = np.nan
    return feature_dict


# Q1b
def impute_missing_values_challenge(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        q1 = np.nanpercentile(column, 25)
        q3 = np.nanpercentile(column, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_column = np.where((column < lower_bound) | (column > upper_bound), np.nan, column)

        mean = np.nanmean(column)
        column[np.isnan(column)] = mean

        X[:, i] = column
    # print(np.array2string(X[170, :], precision=6, suppress_small=True))
    return X


# Q1c
def normalize_feature_matrix_challenge(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        return LogisticRegression(penalty=penalty, solver='liblinear', C=C, fit_intercept=False, random_state=seed, class_weight=class_weight)
    elif loss == "squared_error":
        return KernelRidge(alpha=1/(2*C), kernel = kernel, gamma = gamma)



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
        num_bootstrap_samples = 1000
        sample_size = 1000
        bootstrap_performances = []
        for _ in range(num_bootstrap_samples):
            indices = np.random.choice(len(X), size=sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y_true[indices]
            sample_performance = performance(clf_trained, X_sample, y_sample, metric, False)
            bootstrap_performances.append(sample_performance)
        return (np.median(bootstrap_performances), np.percentile(bootstrap_performances, 2.5), np.percentile(bootstrap_performances, 97.5))
    else:
        y_pred = clf_trained.predict(X)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        if isinstance(clf_trained, KernelRidge):
            for i in range(len(y_pred)):
                if y_pred[i] > 0:
                    if y_true[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if y_true[i] == 1:
                        FN += 1
                    else:
                        TN += 1

        else:
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
        if FP + TP == 0:
            precision = 0
        else:
            precision = TP / (FP + TP)
        if TP + FN == 0:
            sensitivity = 0
        else:
            sensitivity = TP / (TP + FN)
        if metric == "accuracy":
            return (TP + TN) / (TP + TN + FP + FN)
        elif metric == "precision":
            return precision
        elif metric == 'f1_score':
            if precision == 0 or sensitivity == 0 or precision + sensitivity == 0:
                return 0
            else:
                return 2 * (precision * sensitivity) / (precision + sensitivity)
        elif metric == 'auroc':
            y_true = (y_true + 1) / 2
            if isinstance(clf_trained, KernelRidge):
                auroc = roc_auc_score(y_true, y_pred)
            else:
                auroc = roc_auc_score(y_true, clf_trained.decision_function(X))
            return auroc
        elif metric == 'average_precision':
            y_true = (y_true + 1) / 2
            if isinstance(clf_trained, KernelRidge):
                return average_precision_score(y_true, y_pred)
            else:
                return average_precision_score(y_true, clf_trained.decision_function(X))
        elif metric == 'sensitivity':
            return sensitivity
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
        y: (n, ) vector of binary labels {1,-1}
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
    penalties: list[str] = ["l2", "l1"]
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
            print(f"Performance Measures: C={C}, Penalty={penalty}, Mean={mean_performance:.4f} "f"(Min={min_performance:.4f}, Max={max_performance:.4f}) CV Performance")
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_penalty = penalty
    result = (best_C, best_penalty)
    return result


# Q3.2
def select_param_weighted_logreg(
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int64],
        class_weight_range: list[dict[int, float]],
        k: int = 5
) -> dict[int, float]:
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

    best_weight = None
    best_performance = float('-inf')
    for class_weight in class_weight_range:
        clf = get_classifier('logistic','l2',1,class_weight)
        mean_performance = cv_performance(clf, X, y, 'f1_score', k)[0]
        #print(f"Performance Measures: C={C}, Penalty={penalty}, Mean={mean_performance:.4f} "f"(Min={min_performance:.4f}, Max={max_performance:.4f}) CV Performance")
        if mean_performance > best_performance:
            best_performance = mean_performance
            best_weight = class_weight
    return best_weight

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
    best_performance = float('-inf')
    best_C = None
    best_gamma = None
    for C in C_range:
        for gamma in gamma_range:
            clf = get_classifier(loss="squared_error", C=C, kernel='rbf', gamma=gamma)
            mean_performance = cv_performance(clf, X, y, metric, k)[0]
            min_performance = cv_performance(clf, X, y, metric, k)[1]
            max_performance = cv_performance(clf, X, y, metric, k)[2]
            print(f"Performance Measures: Gamma={gamma}, Mean={mean_performance:.4f} "f"(Min={min_performance:.4f}, Max={max_performance:.4f}) CV Performance")
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_gamma = gamma
    # NOTE: This function should be very similar in structure to your implementation of select_param_logreg()
    return (best_C, best_gamma)


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
            clf = get_classifier('logistic', penalty, C)
            # TODO Fit classifier with X and y
            clf.fit(X, y)
            # TODO: Extract learned coefficients/weights from clf into w
            # Note: Refer to sklearn.linear_model.LogisticRegression documentation
            # for attribute containing coefficients/weights of the clf object
            w = clf.coef_


            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0
            non_zero_count = np.sum(w != 0)
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


def select_param_challenge(
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int64],
        metric: str = "accuracy",
        k: int = 5,
        C_range: list[float] = [],
        penalties: list[str] = ["l2", "l1"],
        weight_range: list[dict[int, float]] = []
) -> tuple[float, str, dict[int, float]]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n, ) array of binary labels {1,-1}
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
    best_weight = None
    best_performance = float('-inf')
    for C in C_range:
        for penalty in penalties:
            print(C, penalty)
            for weight in weight_range:
                clf = get_classifier(penalty=penalty, C=C, class_weight=weight)
                performances = cv_performance(clf, X, y, metric, k)
                mean_performance = performances[0]
                if mean_performance > best_performance:
                    best_performance = mean_performance
                    best_C = C
                    best_penalty = penalty
                    best_weight = weight
    final_clf = get_classifier(penalty=best_penalty, C=best_C, class_weight=best_weight)
    print(cv_performance(final_clf, X, y, 'f1_score', k))
    print(cv_performance(final_clf, X, y, 'auroc', k))
    result = (best_C, best_penalty, best_weight)
    return result

def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    #X_train, y_train, X_test, y_test, feature_names = get_train_test_split()
    '''

    for i in range(X_train.shape[1]):
        column = X_train[:, i]
        print(f"{feature_names[i]}: mean={np.mean(column):.6f}, IQR={np.percentile(column, 75) - np.percentile(column, 25):.6f}")
    '''

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    '''
    selected_params = None
    C_range = [10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100, 1000]
    for metric in metric_list:
        print(metric)
        params = select_param_logreg(X_train, y_train, metric, 5, C_range)
        if metric == 'auroc':
            selected_params = params
    clf = get_classifier("logistic", selected_params[1], selected_params[0])
    '''
    '''
    clf = get_classifier("logistic", 'l1', 1)
    clf.fit(X_train, y_train)

    theta = clf.coef_[0]
    print(theta)
    positive = []
    negative = []
    for i in range(len(theta)):
        heapq.heappush(negative, (theta[i], feature_names[i]))
        heapq.heappush(positive, (-theta[i], feature_names[i]))
    for i in range(4):
        print(heapq.heappop(negative))
        positive_feature = heapq.heappop(positive)
        print(-positive_feature[0], positive_feature[1])
    '''
    '''
    for metric in metric_list:
        result = performance(clf, X_test, y_test, metric, True)
        print(f"{metric}: Median = {result[0]:.6f}, 95% CI = [{result[1]:.6f}, {result[2]:.6f}]")

    plot_weight(X_train, y_train, C_range, ['l2', 'l1'])
    '''
    '''
    weighted_clf = get_classifier(C=1, penalty='l2', class_weight={-1: 1, 1: 50})
    weighted_clf.fit(X_train, y_train)
    for metric in metric_list:
        result = performance(weighted_clf, X_test, y_test, metric, True)
        print(f"{metric}: Median = {result[0]:.6f}, 95% CI = [{result[1]:.6f}, {result[2]:.6f}]")
    '''

    class_weight_range = []
    for i, j in zip(range(1, 51, 5), range(50, 0, -5)):
        class_weight_range.append({-1: i, 1: j})
    '''
    class_weight = {-1: 1, 1: 50}
    #class_weight = select_param_weighted_logreg(X_train, y_train, class_weight_range)
    print(class_weight)
    weighted_clf = get_classifier(C=1, penalty='l2', class_weight=class_weight)
    weighted_clf.fit(X_train, y_train)
    for metric in metric_list:
        result = performance(weighted_clf, X_test, y_test, metric, True)
        print(f"{metric}: Median = {result[0]:.6f}, 95% CI = [{result[1]:.6f}, {result[2]:.6f}]")
    '''
    '''
    clf_1 = get_classifier('logistic', 'l2', 1, {-1: 1, 1: 1})
    clf_5 = get_classifier('logistic', 'l2', 1, {-1: 1, 1: 5})
    clf_1.fit(X_train, y_train)
    clf_5.fit(X_train, y_train)
    # Get decision scores
    y_scores_1 = clf_1.decision_function(X_test)  # For Wn=1, Wp=1
    y_scores_5 = clf_5.decision_function(X_test)  # For Wn=1, Wp=5

    # Calculate ROC curve and AUC for both models
    fpr_1, tpr_1, _ = roc_curve(y_test, y_scores_1)
    fpr_5, tpr_5, _ = roc_curve(y_test, y_scores_5)

    # Calculate AUC
    auc_1 = auc(fpr_1, tpr_1)
    auc_5 = auc(fpr_5, tpr_5)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr_1, tpr_1, color='blue', lw=2, label=f'Wn=1, Wp=1 (AUC = {auc_1:.2f})')
    plt.plot(fpr_5, tpr_5, color='red', lw=2, label=f'Wn=1, Wp=5 (AUC = {auc_5:.2f})')

    # Plot settings
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    '''
    '''
    lrg = LogisticRegression(penalty='l2', C=1, fit_intercept=False, random_state=seed)
    kr = KernelRidge(alpha=1/2, kernel='linear')
    lrg.fit(X_train, y_train)
    kr.fit(X_train, y_train)
    for metric in metric_list:
        print(metric)
        lrg_result = performance(lrg, X_test, y_test, metric)
        print(lrg_result)
        kr_result = performance(kr, X_test, y_test, metric)
        print(kr_result)

    #gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
    #params = select_param_RBF(X_train, y_train, 'auroc', 5, [1], gamma_list)

    C_range = [10 ** -2, 10 ** -1, 1, 10]
    gamma_list = [0.01, 0.1, 1, 10, 100]
    params = select_param_RBF(X_train, y_train, 'auroc', 5, C_range, gamma_list)
    print(params)
    clf = get_classifier(loss="squared_error", C=params[0], kernel='rbf', gamma=params[1])
    clf.fit(X_train, y_train)
    for metric in metric_list:
        print(metric)
        result = performance(clf, X_test, y_test, metric)
        print(result)

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recommended that you create functions for each
    #       sub-question/question to organize your code!
    
    '''

    X_train, y_train, X_heldout, feature_names = get_challenge_data()
    C_range = [0.001, 0.01, 0.1, 1, 10]
    penalties = ['l2', 'l1']
    challenge_params = select_param_challenge(X_train, y_train, 'auroc', C_range=C_range, penalties=penalties, weight_range=class_weight_range)

    print(challenge_params)
    challenge_clf = get_classifier('logistic', challenge_params[1], challenge_params[0], challenge_params[2])
    challenge_clf.fit(X_train, y_train)
    y_score = challenge_clf.decision_function(X_heldout)
    y_label = challenge_clf.predict(X_heldout)
    y_label = y_label.astype(int)
    generate_challenge_labels(y_label, y_score, 'wndu')

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    challenge_clf.fit(X_train_c, y_train_c)
    y_pred = challenge_clf.predict(X_test_c)
    confusion_mat = confusion_matrix(y_test_c, y_pred)
    print(confusion_mat)

    # Read challenge data
    # TODO: Question 5: Apply a classifier to holdout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()



if __name__ == "__main__":
    main()