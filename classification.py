# Classification problem: estimate the total number conflicts between UAVs given the provided features
# What we have to do?
# Predict the number of collision between the UAVs given the previous features. You will have 4 classes and the dataset is unbalanced
# The dataset is provided in the file "train_set.tsv"
# We have 7 columns in the dataset:
# 0. UAV_i_track: clockwise angle from north between the ith UAV and its target (0, 2*pi)
# 1. UAV_i_x: x coordinate of the ith UAV in meters
# 2. UAV_i_y: y coordinate of the ith UAV in meters
# 3. UAV_i_vx: x velocity of the ith UAV in m/s
# 4. UAV_i_vy: y velocity of the ith UAV in m/s
# 5. UAV_i_target_x: x coordinate of the ith UAV target in meters
# 6. UAV_i_target_y: y coordinate of the ith UAV target in meters

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from utils import Approaches, custom_oversampling, normalize_data


file = "./data/train_set.tsv"
seed = 1
test_size = 0.25
oversampling_approach: Approaches = "CUSTOM"
usingAngles = False

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

    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -2]

    # Since the dataset is unbalanced, we have to balance it
    print(f" - Balancing the dataset ({oversampling_approach})...")
    if oversampling_approach == "SMOTE":
        smote = SMOTE(random_state=seed, k_neighbors=2)
        X, y = smote.fit_resample(X, y)
    elif oversampling_approach == "RANDOM_OVER_SAMPLING":
        ros = RandomOverSampler(random_state=seed)
        X, y = ros.fit_resample(X, y)
    elif oversampling_approach == "CUSTOM":
        X, y = custom_oversampling(X, y)
        
    # Remove the columns "UAV_i_track" if not using angles
    if not usingAngles:
        # We want to remove UAV_i_track
        print(" - Removing the columns 'UAV_i_track'...")
        X = X.drop(columns=[f"UAV_{i}_track" for i in range(1, 6)])
        
    # Normalize the features
    print(" - Normalizing the features...")
    X = normalize_data(X)

    # Split the dataset into training and test set
    print(f" - Splitting the dataset into training ({100-test_size*100}%) and test set ({test_size*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def logistic_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Train a logistic regression model and evaluate it.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    print(" - Training a logistic regression model...")
    # Use sklearn to train a logistic regression model
    classifier = LogisticRegression(random_state=seed, max_iter=10000)
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print('Logistic Regression')
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))
    print('Cross Validation')
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    scores = cross_val_score(classifier, X_train, y_train, cv=kfold)
    print(scores)
    print('Mean Accuracy')
    print(scores.mean())
    print('Standard Deviation')
    print(scores.std())

def main():
    """
    Classification problem: estimate the total number conflicts between UAVs given the provided features.
    """
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Train the model using 4 different classifiers
    # 1. Logistic Regression
    logistic_regression(X_train, X_test, y_train, y_test)
    # 2. K-Nearest Neighbors
    # 3. Support Vector Machine
    # 4. Decision Tree

if __name__ == '__main__':
    main()
