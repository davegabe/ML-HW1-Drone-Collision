from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from utils import normalize_data, normalize_data_rect
pd.set_option('mode.chained_assignment', None)

file = "./data/train_set.tsv"
seed = 1
test_size = 0.2
use_angle = True

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset from the file.
    
    Returns:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Load the dataset
    print(" - Loading the dataset...")
    dataset = pd.read_csv(file, sep='\t', header=0)

    
    # Normalize the dataset
    dataset = normalize_data(dataset.iloc[:, :]) 

    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -1]

    # Remove the columns "UAV_i_track" if not using angles
    if not use_angle:
        X = X.drop(columns=[f"UAV_{i}_track" for i in range(1, 6)])

    # Split the dataset into training and test set
    print(f" - Splitting the dataset into training ({100-test_size*100}%) and test set ({test_size*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def linear_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """
    Train a linear regression model.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Train the model
    print(" - Training the model...")
    model = LogisticRegression(random_state=seed, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print(" - Evaluating the model...")
    y_pred = model.predict(X_test)
    print(f" - Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f" - R2 score: {r2_score(y_test, y_pred):.2f}")
    


def random_forest_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """
    Train a random forest regression model.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Train the model
    print(" - Training the model...")
    model = RandomForestRegressor(random_state=seed, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    print(" - Evaluating the model...")
    y_pred = model.predict(X_test)
    print(f" - Mean squared error: {mean_squared_error(y_test, y_pred):.5f}")
    print(f" - R2 score: {r2_score(y_test, y_pred):.5f}")

    # Plot predicted vs actual values, use reset_index
    plt.scatter(y_test.reset_index(drop=True), y_pred, color='black')
    plt.title("Predicted vs actual values")
    plt.show()

def support_vector_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """
    Train a support vector regression model.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Train the model
    print(" - Training the model...")
    model = SVR(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)

    # Evaluate the model
    print(" - Evaluating the model...")
    y_pred = model.predict(X_test)
    print(f" - Mean squared error: {mean_squared_error(y_test, y_pred):.5f}")
    print(f" - R2 score: {r2_score(y_test, y_pred):.5f}")

    # Plot predicted vs actual values
    plt.scatter(y_test, y_pred, color='black')
    plt.title("Predicted vs actual values")
    plt.show()

def main():
    """
    Regression problem: predict the minimum Closest Point of Approach (CPA) among all the possible pairs of UAVs.
    """
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # We have the label CPA in meters, continuous values
    # Train the model using 3 different algorithms
    # # 1. Linear regression
    # linear_regression(X_train, X_test, y_train, y_test)
    # 2. Random forest regression
    random_forest_regression(X_train, X_test, y_train, y_test)
    # 3. Support vector regression
    support_vector_regression(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
