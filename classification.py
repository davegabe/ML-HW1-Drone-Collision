import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils import Approaches, custom_oversampling_all, normalize_data


file = "./data/train_set.tsv"
test_size = 0.2
oversampling_approach: Approaches = "CUSTOM"


def load_dataset(seed: int, use_angle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset from the file.

    Returns:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Load the dataset
    dataset = pd.read_csv(file, sep='\t', header=0)

    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -2]

    # Remove the columns "UAV_i_track" if not using angles
    if not use_angle:
        X = X.drop(columns=[f"UAV_{i}_track" for i in range(1, 6)])

    # Normalize the features
    X = normalize_data(X)

    # Split the dataset into training and test set, stratified by the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    if oversampling_approach == "SMOTETomek":
        smote = SMOTE(random_state=seed)
        smotetomek = SMOTETomek(random_state=seed, smote=smote)
        X_train, y_train = smotetomek.fit_resample(X_train, y_train)
    elif oversampling_approach == "CUSTOM":
        X_train, y_train = custom_oversampling_all(X_train, y_train, 50)

    return X_train, X_test, y_train, y_test


def random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a random forest model and evaluate it.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Use sklearn to train a random forest model
    classifier = RandomForestClassifier(n_estimators=50, random_state=seed, criterion='gini', max_features='sqrt')
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def svm(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a SVM model and evaluate it.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Use sklearn to train a SVM model
    classifier = SVC(kernel='rbf', random_state=seed, C=10, gamma='scale')
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def logistic_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a logistic regression model and evaluate it.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Use sklearn to train a logistic regression model
    classifier = LogisticRegression(random_state=seed, max_iter=1000, solver='liblinear', penalty='l1', C=5)
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def gaussian_naive_bayes(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a Gaussian Naive Bayes model and evaluate it.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Use sklearn to train a Gaussian Naive Bayes model
    classifier = GaussianNB(var_smoothing=1e-6)
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def main(seed: int, use_angle: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Classification problem: estimate the total number conflicts between UAVs given the provided features.
    """
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset(seed, use_angle)

    # Dictionary to store the results
    predict = dict()
    
    # Random Forest
    predict["Random Forest"] = random_forest(X_train, X_test, y_train, y_test, seed)
    # SVM
    predict["SVM"] = svm(X_train, X_test, y_train, y_test, seed)
    # Logistic Regression
    predict["Logistic Regression"] = logistic_regression(X_train, X_test, y_train, y_test, seed)
    # Gaussian Naive Bayes
    predict["Gaussian Naive Bayes"] = gaussian_naive_bayes(X_train, X_test, y_train, y_test)

    return predict


if __name__ == '__main__':
    result = main(seed=42, use_angle=False)
    for key, value in result.items():
        print(f"{key}:")
        print(f" - accuracy: {accuracy_score(value[0], value[1])}")
        print(f" - precision: {precision_score(value[0], value[1], average='weighted', zero_division=0)}")
        print(f" - recall: {recall_score(value[0], value[1], average='weighted', zero_division=0)}")
        print(f" - f1: {f1_score(value[0], value[1], average='weighted', zero_division=0)}")
        print(f" - confusion matrix: {confusion_matrix(value[0], value[1])}")
