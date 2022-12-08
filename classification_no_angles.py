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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.naive_bayes import GaussianNB
from utils import Approaches, custom_oversampling, custom_oversampling_minority, normalize_data_rect


file = "./data/train_set.tsv"
test_size = 0.2
oversampling_approach: Approaches = "SMOTETomek"
pre_over_sampling_approach: Approaches = "CUSTOM"


def load_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Split the dataset into features and labels (without UAV_i_track)
    X = dataset.iloc[:, 1:-2]
    y = dataset.iloc[:, -2]
        
    # Normalize the features
    print(" - Normalizing the features...")
    X = normalize_data_rect(X)

    # Split the dataset into training and test set, stratified by the labels
    print(f" - Splitting the dataset into training ({100-test_size*100}%) and test set ({test_size*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # Since the dataset is unbalanced, we have to balance it
    print(f" - Balancing the dataset ({oversampling_approach})...")
    if pre_over_sampling_approach == "CUSTOM":
        X_train, y_train = custom_oversampling_minority(X_train, y_train, 10)
    elif pre_over_sampling_approach == "RandomOverSampler":
        randomversampler = RandomOverSampler(random_state=seed, sampling_strategy='minority')
        X_train, y_train = randomversampler.fit_resample(X_train, y_train)

    if oversampling_approach == "SMOTETomek":
        print(" - SMOTETomek")
        smote = SMOTETomek(random_state=seed, smote=SMOTE(random_state=seed), sampling_strategy='minority')
        # smote = SMOTE(random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif oversampling_approach == "CUSTOM":
        X_train, y_train = custom_oversampling(X_train, y_train)
        pass
    
    return X_train, X_test, y_train, y_test

def logistic_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> None:
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
    classifier = LogisticRegression(random_state=seed, max_iter=1000)
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print('Logistic Regression')
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    # use confusion matrix from seaborn
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))
    print('Cross Validation')

def gaussian_naive_bayes(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> None:
    """
    Train a Gaussian Naive Bayes model and evaluate it.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    print(" - Training a Gaussian Naive Bayes model...")
    # Use sklearn to train a Gaussian Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print('Gaussian Naive Bayes')
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))
    print('Cross Validation')

def random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a random forest model and evaluate it.
    
    Args:
        X_train: training set
        X_test: test set
        y_train: training labels
        y_test: test labels
    """
    print(" - Training a random forest model...")
    # Use sklearn to train a random forest model
    classifier = RandomForestClassifier(n_estimators=50, random_state=seed)
    classifier.fit(X_train, y_train)
    # confusion matrix
    y_pred = classifier.predict(X_test)
    # Evaluate the model
    y_pred = classifier.predict(X_test)

    return y_test, y_pred

def main(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Classification problem: estimate the total number conflicts between UAVs given the provided features.
    """
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset(seed)

    predict = dict()
    # Train the model using 4 different classifiers for imbalanced data
    # 1. Random Forest
    predict["Random Forest"] = random_forest(X_train, X_test, y_train, y_test, seed)
    # 2. SVM
    # svm(X_train, X_test, y_train, y_test, seed)
    # 3. Logistic Regression
    # logistic_regression(X_train, X_test, y_train, y_test, seed)
    # 4. Gaussian Naive Bayes
    # gaussian_naive_bayes(X_train, X_test, y_train, y_test, seed)

    return predict


if __name__ == '__main__':
    main(seed=42)