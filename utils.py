from typing import Literal
import numpy as np
import pandas as pd
from scipy import sparse

Approaches = Literal['NONE', 'SMOTE', 'RANDOM_OVER_SAMPLING', 'SMOTE+RANDOM_OVER_SAMPLING', 'CUSTOM']

def normalize_data(X: pd.Series) -> pd.Series:
    """
    Normalize the data. 
    Coordinates are normalized between 0 and 1.
    Aspects ratios are kept, so the angles are not changed.

    Args:
        X: dataset

    Returns:
        X: normalized dataset
    """
    # Normalize the coordinates of certain columns (UAV_i_x, UAV_i_y, UAV_i_vx, UAV_i_vy, UAV_i_target_x, UAV_i_target_y)
    for row in range(len(X)):  # for each row in the dataset
        # x coordinates of the UAVs and their targets
        x_coordinates = np.array([])
        # y coordinates of the UAVs and their targets
        y_coordinates = np.array([])
        for i in range(1, 6):  # for each UAV
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
            X[f"UAV_{j}_x"][row] = (X[f"UAV_{j}_x"][row] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_y"][row] = (X[f"UAV_{j}_y"][row] - min_y) / (max_y - min_y)
            X[f"UAV_{j}_target_x"][row] = (X[f"UAV_{j}_target_x"][row] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_target_y"][row] = (X[f"UAV_{j}_target_y"][row] - min_y) / (max_y - min_y)
            X[f"UAV_{j}_vx"][row] = X[f"UAV_{j}_vx"][row] / (max_x - min_x)
            X[f"UAV_{j}_vy"][row] = X[f"UAV_{j}_vy"][row] / (max_y - min_y)

        # If has "min_CPA" column, we normalize it
        if "min_CPA" in X.columns:
            X["min_CPA"][row] = X["min_CPA"][row] / (max_x - min_x)
    return X

def normalize_data_rect(X: pd.Series) -> pd.Series:
    """
    Normalize the data.
    Coordinates are normalized between 0 and 1.
    Aspects ratios are not kept, so the angles are changed.

    Args:
        X: dataset

    Returns:
        X: normalized dataset
    """
    # Normalize the coordinates of certain columns (UAV_i_x, UAV_i_y, UAV_i_vx, UAV_i_vy, UAV_i_target_x, UAV_i_target_y)
    for row in range(len(X)):  # for each row in the dataset
        # x coordinates of the UAVs and their targets
        x_coordinates = np.array([])
        # y coordinates of the UAVs and their targets
        y_coordinates = np.array([])
        for i in range(1, 6):  # for each UAV
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

        # We normalize the coordinates of the UAVs and their targets using the min and max values
        for j in range(1, 6):
            X[f"UAV_{j}_x"][row] = (X[f"UAV_{j}_x"][row] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_y"][row] = (X[f"UAV_{j}_y"][row] - min_y) / (max_y - min_y)
            X[f"UAV_{j}_target_x"][row] = (X[f"UAV_{j}_target_x"][row] - min_x) / (max_x - min_x)
            X[f"UAV_{j}_target_y"][row] = (X[f"UAV_{j}_target_y"][row] - min_y) / (max_y - min_y)
            X[f"UAV_{j}_vx"][row] = X[f"UAV_{j}_vx"][row] / (max_x - min_x)
            X[f"UAV_{j}_vy"][row] = X[f"UAV_{j}_vy"][row] / (max_y - min_y)

        # If has "min_CPA" column, we normalize it
        if "min_CPA" in X.columns:
            X["min_CPA"][row] = X["min_CPA"][row] / (max_x - min_x)
    return X

def normalize_all(X: pd.Series) -> pd.Series:
    """
    Normalize the data. 
    Coordinates are normalized between 0 and 1.
    Aspects ratios are kept, so the angles are not changed.

    Args:
        X: dataset

    Returns:
        X: normalized dataset
    """
    # Normalize all the features of the dataset
    X = X - X.min() / (X.max() - X.min())
    return X

def custom_oversampling(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Custom oversampling for all classes except the majority class.

    Args:
        X: dataset
        y: labels

    Returns:
        X: oversampled dataset
        y: oversampled labels
    """
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]
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
            # Create synthetic samples
            for j in range(number_of_samples_to_create):
                # Take a random sample from the current class
                rnd = np.random.randint(0, current_class_count)
                sample = X[y == i].iloc[rnd].copy(deep=True)
                # For each drone in the sample
                for k in range(1, 6):
                    # Take all the features of the drone
                    drone_x = sample[f"UAV_{k}_x"]
                    drone_y = sample[f"UAV_{k}_y"]
                    vx = sample[f"UAV_{k}_vx"]
                    vy = sample[f"UAV_{k}_vy"]
                    # Randomly choose a time step
                    time_step = np.random.uniform(0.1, 10)
                    # Move the drone backwards
                    drone_x -= vx * time_step
                    drone_y -= vy * time_step
                    # Update the features of the drone
                    sample[f"UAV_{k}_x"] = drone_x
                    sample[f"UAV_{k}_y"] = drone_y
                # Add the sample to the dataset
                X_resampled.append(sample)
                y_resampled.append(i)
    # Return the oversampled dataset
    if sparse.issparse(X):
        X_resampled = sparse.vstack(X_resampled, format=X.format)
    else:
        X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    # Convert to pandas Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)
    return X_resampled, y_resampled

def custom_oversampling_all(X: pd.DataFrame, y: pd.Series, count: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Custom oversampling for all classes add a specific number of samples.

    Args:
        X: dataset
        y: labels

    Returns:
        X: oversampled dataset
        y: oversampled labels
    """
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]
    # For each other class
    for i in range(len(np.bincount(y))):
        # Find the number of samples in the current class
        current_class_count = np.bincount(y)[i]
        # Create synthetic samples
        for j in range(count):
            # Take a random sample from the current class
            rnd = np.random.randint(0, current_class_count)
            sample = X[y == i].iloc[rnd].copy(deep=True)
            # For each drone in the sample
            for k in range(1, 6):
                # Take all the features of the drone
                drone_x = sample[f"UAV_{k}_x"]
                drone_y = sample[f"UAV_{k}_y"]
                vx = sample[f"UAV_{k}_vx"]
                vy = sample[f"UAV_{k}_vy"]
                # Randomly choose a time step
                time_step = np.random.uniform(0.1, 10)
                # Move the drone backwards
                drone_x -= vx * time_step
                drone_y -= vy * time_step
                # Update the features of the drone
                sample[f"UAV_{k}_x"] = drone_x
                sample[f"UAV_{k}_y"] = drone_y
            # Add the sample to the dataset
            X_resampled.append(sample)
            y_resampled.append(i)
    # Return the oversampled dataset
    if sparse.issparse(X):
        X_resampled = sparse.vstack(X_resampled, format=X.format)
    else:
        X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    # Convert to pandas Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)
    return X_resampled, y_resampled


def custom_oversampling_minority(X: pd.DataFrame, y: pd.Series, count:int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Custom oversampling for the minority class.

    Args:
        X: dataset
        y: labels

    Returns:
        X: oversampled dataset
        y: oversampled labels
    """
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]
    # Find minority class
    min_class = np.argmin(np.bincount(y))
    # Create synthetic samples
    for i in range(count):
        # Take a random sample from the current class
        rnd = np.random.randint(0, len(X[y == min_class]))
        sample = X[y == min_class].iloc[rnd].copy(deep=True)
        # For each drone in the sample
        for k in range(1, 6):
            # Take all the features of the drone
            drone_x = sample[f"UAV_{k}_x"]
            drone_y = sample[f"UAV_{k}_y"]
            vx = sample[f"UAV_{k}_vx"]
            vy = sample[f"UAV_{k}_vy"]
            # Randomly choose a time step
            time_step = np.random.uniform(0.1, 10)
            # Move the drone backwards
            drone_x -= vx * time_step
            drone_y -= vy * time_step
            # Update the features of the drone
            sample[f"UAV_{k}_x"] = drone_x
            sample[f"UAV_{k}_y"] = drone_y
        # Add the sample to the dataset
        X_resampled.append(sample)
        y_resampled.append(min_class)
    # Return the oversampled dataset
    if sparse.issparse(X):
        X_resampled = sparse.vstack(X_resampled, format=X.format)
    else:
        X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    # Convert to pandas Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)
    return X_resampled, y_resampled