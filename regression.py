from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from utils import normalize_data

pd.set_option('mode.chained_assignment', None)

file = "./data/train_set.tsv"
test_size = 0.2


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

    # Normalize the dataset
    dataset = normalize_data(dataset.iloc[:, :])

    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -1]

    # Remove the columns "UAV_i_track" if not using angles
    if not use_angle:
        X = X.drop(columns=[f"UAV_{i}_track" for i in range(1, 6)])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def random_forest_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a random forest regression model.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Train the model
    model = RandomForestRegressor(random_state=seed, n_estimators=300, criterion='poisson', max_features='sqrt')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    return y_test, y_pred


def support_vector_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a support vector regression model.

    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Train the model
    model = SVR(kernel='rbf', gamma='scale', C=7)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    return y_test, y_pred


def main(seed: int, use_angle: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Regression problem: predict the minimum Closest Point of Approach (CPA) among all the possible pairs of UAVs.
    """
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset(seed, use_angle)

    # Dictionary to store the predictions
    predict = dict()

    # Random forest regression
    predict["Random Forest Regression"] = random_forest_regression(X_train, X_test, y_train, y_test, seed)
    
    # Support vector regression
    predict["Support Vector Regression"] = support_vector_regression(X_train, X_test, y_train, y_test, seed)

    return predict


if __name__ == '__main__':
    result = main(seed=42, use_angle=False)
    for key, value in result.items():
        print(f"{key}:")
        print(f" - Mean squared error: {mean_squared_error(value[0], value[1]):.5f}")
        print(f" - R2 score: {r2_score(value[0], value[1]):.5f}")
