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
from typing import Literal

seed = 1
time_step = 0.01
test_size = 0.25
Approaches = Literal['NONE', 'SMOTE', 'RANDOM_OVER_SAMPLING', 'SMOTE+RANDOM_OVER_SAMPLING', 'CUSTOM']
oversampling_approach: Approaches = "NONE"
usingAngles = False

# # DataField is a class that represents a field in the dataset
# DataField = Literal[
#     'UAV_1_track', 'UAV_1_x', 'UAV_1_y', 'UAV_1_vx', 'UAV_1_vy', 'UAV_1_target_x', 'UAV_1_target_y',
#     'UAV_2_track', 'UAV_2_x', 'UAV_2_y', 'UAV_2_vx', 'UAV_2_vy', 'UAV_2_target_x', 'UAV_2_target_y',
#     'UAV_3_track', 'UAV_3_x', 'UAV_3_y', 'UAV_3_vx', 'UAV_3_vy', 'UAV_3_target_x', 'UAV_3_target_y',
#     'UAV_4_track', 'UAV_4_x', 'UAV_4_y', 'UAV_4_vx', 'UAV_4_vy', 'UAV_4_target_x', 'UAV_4_target_y',
#     'UAV_5_track', 'UAV_5_x', 'UAV_5_y', 'UAV_5_vx', 'UAV_5_vy', 'UAV_5_target_x', 'UAV_5_target_y',
# ]

def create_paths_from_dataset(X: np.ndarray) -> np.ndarray:
    """
    Create paths from the dataset.
    
    Args:
        X: dataset
    
    Returns:
        paths: paths
    """
    # Each row of the dataset represents the state of the UAVs at a given time
    # Having their position, velocity and the angle between the UAV and its target
    # We can use this information to predict the number of collision between the UAVs
    # We create for each drone a list of position in the form (x, y)

    # Create paths for each drone of each row of the dataset
    paths: list[list[list[tuple[float, float]]]] = [[[] for _ in range(5)] for _ in range(len(X))]
    # print(np.cos(np.pi/2-np.pi/2))
    # return
    # For each row of the dataset
    for row in range(1):
        # For each drone
        for drone in range(5):
            print("Drone: ", drone)
            # Get the clockwise angle from north between the ith UAV and its target (0, 2*pi)
            angle = X[row, drone * 7]
            # Get the position of the drone
            x = X[row, drone * 7 + 1]
            y = X[row, drone * 7 + 2]
            # Get the velocity of the drone
            vx = X[row, drone * 7 + 3]
            vy = X[row, drone * 7 + 4]
            # Get the target of the drone
            target_x = X[row, drone * 7 + 5]
            target_y = X[row, drone * 7 + 6]
            time = 0
            paths[row][drone].append((x, y))
            # Compute tha paths, since we normalize the dataset, we have domain of x and y in [0, 1]
            while 1>=x>=0 and 1>=y>=0:
                # Compute the new position of the drone
                x += vx * np.cos(angle) * time_step
                y += vy * np.sin(angle) * time_step
                # Add the new position to the path of the drone
                paths[row][drone].append((x, y))
                time += time_step
            # print(paths[row][drone])
            # print("Target: ", (target_x, target_y))
    return paths

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize the data.

    Args:
        X: dataset
    
    Returns:
        X: normalized dataset
    """
    # Normalize the coordinates of certain columns (UAV_i_x, UAV_i_y, UAV_i_vx, UAV_i_vy, UAV_i_target_x, UAV_i_target_y)
    for row in range(len(X)): # for each row in the dataset
        x_coordinates = np.array([]) # x coordinates of the UAVs and their targets
        y_coordinates = np.array([]) # y coordinates of the UAVs and their targets
        for i in range(1, 6): # for each UAV
            # We add the x and y coordinates of the UAV and its target
            x_coordinates = np.append(x_coordinates, X[f"UAV_{i}_x"][row])
            x_coordinates = np.append(x_coordinates, X[f"UAV_{i}_target_x"][row])
            y_coordinates = np.append(y_coordinates, X[f"UAV_{i}_y"][row])
            y_coordinates = np.append(y_coordinates, X[f"UAV_{i}_target_y"][row])

        # We find th min x and y coordinates of the UAVs and their targets
        min_x = min(x_coordinates)
        min_y = min(y_coordinates)
        # We find the max x and y coordinates of the UAVs and their targets
        max_x = max(x_coordinates)
        max_y = max(y_coordinates)

        # Since we want to keep an aspect ratio of 1 (to keep angles), we have to find the max value between max_x and max_y
        max_x = max(max_x, max_y)
        max_y = max_x
        # And we have to find the min value between min_x and min_y
        min_x = min(min_x, min_y)
        min_y = min_x

        # We normalize the coordinates of the UAVs and their targets using the min and max values
        for j in range(1, 6):
            X[f"UAV_{j}_x"] = (X[f"UAV_{j}_x"] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_y"] = (X[f"UAV_{j}_y"] - min_y) / (max_y - min_y)
            X[f"UAV_{j}_target_x"] = (X[f"UAV_{j}_target_x"] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_target_y"] = (X[f"UAV_{j}_target_y"] - min_y) / (max_y - min_y)
    return X

def custom_oversampling(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom oversampling.
    
    Args:
        X: dataset
        y: labels
    
    Returns:
        X: oversampled dataset
        y: oversampled labels
    """
    # Find the number of samples in the majority class
    majority_class = np.argmax(np.bincount(y))
    majority_class_count = np.bincount(y)[majority_class]
    # For each other class (except the majority class)
    for i in range(len(np.bincount(y))):
        if i != majority_class:
            # Find the number of samples in the current class
            current_class_count = np.bincount(y)[i]
            # Find the number of samples to create
            number_of_samples_to_create = majority_class_count - current_class_count
            # Create synthetic samples using create_paths_from_dataset
            for j in range(number_of_samples_to_create):
                pass

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset from the file "train_set.tsv".
    
    Returns:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    # Load the dataset
    dataset = pd.read_csv('train_set.tsv', sep='\t', header=0)
    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -2]
    if not usingAngles:
        # We want to remove UAV_i_track
        X = X.drop(columns=[f"UAV_{i}_track" for i in range(1, 6)])
        
    # Normalize the features
    X = normalize_data(X)

    # Since the dataset is unbalanced, we have to balance it
    if oversampling_approach == "SMOTE":
        smote = SMOTE(random_state=seed, k_neighbors=2)
        X, y = smote.fit_resample(X, y)
    elif oversampling_approach == "RANDOM_OVER_SAMPLING":
        ros = RandomOverSampler(random_state=seed)
        X, y = ros.fit_resample(X, y)
    elif oversampling_approach == "CUSTOM":
        X, y = custom_oversampling(X, y)

    # Split the dataset into training and test set
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
    # Use sklearn to train a logistic regression model
    classifier = LogisticRegression(random_state=seed, max_iter=100000)
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
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset()
    # print(X_train[0])

    # Train the model using 4 different classifiers
    # 1. Logistic Regression
    logistic_regression(X_train, X_test, y_train, y_test)
    # 2. K-Nearest Neighbors
    # 3. Support Vector Machine
    # 4. Decision Tree

if __name__ == '__main__':
    main()
