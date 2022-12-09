import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from classification import load_dataset


def main_classification():
    """
    Launch grid search on classification model.
    """
    seed = 42
    use_angle = False
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset(seed, use_angle)
    # Define the models and parameters to explore
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=seed),
            'params': {
                'n_estimators': [i*5 for i in range(1, 20)],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': [None, 'sqrt', 'log2']
            }
        },
        'SVM': {
            'model': SVC(random_state=seed),
            'params': {
                'C': [0.5 * i for i in range(1, 15)],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']

            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=seed, max_iter=3000),
            'params': {
                'C': [0.5 * i for i in range(1, 15)],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky']
            }
        },
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            }
        }
    }
    # Test GridSearchCV on each model
    score = make_scorer(f1_score, average='weighted', zero_division=0)
    for name, model in models.items():
        print("#"*25)
        print(f"Grid search on {name}")
        clf = GridSearchCV(model['model'], model['params'], scoring=score, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        print(f" - best parameters: {clf.best_params_}")
        print(f" - best score: {clf.best_score_}")
        # Evaluate the model
        y_pred = clf.predict(X_test)
        print(f" - accuracy: {accuracy_score(y_test, y_pred)}")
        print(f" - f1: {f1_score(y_test, y_pred, average='weighted', zero_division=0)}")
        print("#"*25)


def main_regression():
    """
    Launch grid search on regression model.
    """
    seed = 42
    use_angle = False
    # Set the seed
    np.random.seed(seed)
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset(seed, use_angle)
    # Define the models and parameters to explore
    models = {
        'Support Vector Regression': {
            'model': SVR(),
            'params': {
                'C': [0.5 * i for i in range(1, 15)],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        },
        'Gradient Boosting Regression': {
            'model': GradientBoostingRegressor(random_state=seed),
            'params': {
                'n_estimators': [i*25 for i in range(1, 15)],
                'learning_rate': [0.05 * i for i in range(1, 20)],
                'loss': ['huber', 'quantile', 'squared_error', 'absolute_error'],
                'max_depth': [i for i in range(1, 10)],
                'max_features': [None, 'sqrt', 'log2']
            }
        },
    }
    # Test GridSearchCV on each model
    score = 'r2'
    for name, model in models.items():
        print("#"*25)
        print(f"Grid search on {name}")
        clf = GridSearchCV(model['model'], model['params'], scoring=score, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        print(f" - best parameters: {clf.best_params_}")
        print(f" - best score: {clf.best_score_}")
        print("#"*25)


if __name__ == "__main__":
    main_classification()
    main_regression()
