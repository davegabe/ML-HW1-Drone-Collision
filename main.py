from matplotlib import pyplot as plt
import numpy as np
from classification import main as classification
from regression import main as regression
from multiprocessing import Pool
import builtins
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, precision_score, r2_score, recall_score
import seaborn as sns


def mute():
    """
    Mute the print function.
    """
    builtins.print = lambda *args, **kwargs: None


def main_classification():
    """
    Launch classification model on different seeds and plot the results.
    """
    print("#"*10 + " Classification " + "#"*10)
    # List of predictions for each seed, each prediction is a dictionary with the name of the classifier as key and (y_test, y_pred) as value
    predicts: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    n_processes = 10
    n_seed = 30
    use_angle = False

    # Launch the classification model on different seeds
    with Pool(n_processes, initializer=mute) as p:
        predicts = p.starmap(classification, [(seed, use_angle) for seed in range(n_seed)])

    # For each classifier, compute the average accuracy, precision, recall and f1-score
    for classifier in predicts[0].keys():
        # List of accuracy, precision, recall and f1-score for each seed
        accuracy: list[float] = []
        precision: list[float] = []
        recall: list[float] = []
        f1: list[float] = []

        for predict in predicts:
            y_test, y_pred = predict[classifier]
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            recall.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        print(f"{classifier}:")
        print(f"\tAccuracy: {np.mean(accuracy)}")
        print(f"\tPrecision: {np.mean(precision)}")
        print(f"\tRecall: {np.mean(recall)}")
        print(f"\tF1-score: {np.mean(f1)}")
        print("")

        # Find the best seed
        best_seed = np.argmax(f1)
        y_test, y_pred = predicts[best_seed][classifier]
        # print(f"\tBest seed: {best_seed}")

        # Plot the confusion matrix for the best seed using seaborn
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
        plt.title(f'Confusion matrix for {classifier}')
        plt.savefig(f'./results/confusion_matrix_{classifier.replace(" ", "_")}.png', dpi=600)
        plt.close()

        # Plot the predict/true distribution for the best seed
        plt.title(f'Prediction for {classifier}')
        plt.xlabel('Sample')
        plt.ylabel('Class')
        plt.yticks(np.arange(0, 5, 1))
        len_arr = len(y_test)
        plt.plot(range(len_arr), y_test, label='Real')
        plt.plot(range(len_arr), y_pred, label='Predicted')
        plt.legend()
        filename = f'./results/prediction_{classifier.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

        # Plot the f1-score for all the seeds with the average 
        plt.title(f'F1-score for {classifier}')
        plt.xlabel('Seeds')
        plt.ylabel('F1-score')
        plt.ylim(0.34, 0.52)
        plt.plot(range(len(f1)), f1, label='F1-score')
        plt.plot(range(len(f1)), [np.mean(f1)]*len(f1), label='Average', linestyle='--')
        plt.legend()
        filename = f'./results/f1_score_{classifier.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

        # Plot the accuracy for all the seeds with the average
        plt.title(f'Accuracy for {classifier}')
        plt.xlabel('Seeds')
        plt.ylabel('Accuracy')
        plt.ylim(0.35, 0.57)
        plt.plot(range(len(accuracy)), accuracy, label='Accuracy')
        plt.plot(range(len(accuracy)), [np.mean(accuracy)]*len(accuracy), label='Average', linestyle='--')
        plt.legend()
        filename = f'./results/accuracy_{classifier.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()


def main_regression():
    """
    Launch regression model on different seeds and plot the results.
    """
    print("#"*10 + " Regression " + "#"*10)
    # List of predictions for each seed, each prediction is a dictionary with the name of the regressor as key and (y_test, y_pred) as value
    predicts: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    n_processes = 10
    use_angle = False

    # Launch the regression model on different seeds
    with Pool(n_processes, initializer=mute) as p:
        predicts = p.starmap(regression, [(seed, use_angle) for seed in range(n_processes)])

    # For each regressor, compute the average accuracy, precision, recall and f1-score
    for regressor in predicts[0].keys():
        # List of accuracy, precision, recall and f1-score for each seed
        mse: list[float] = []
        r2: list[float] = []

        for predict in predicts:
            y_test, y_pred = predict[regressor]
            mse.append(mean_squared_error(y_test, y_pred))
            r2.append(r2_score(y_test, y_pred))

        print(f"{regressor}:")
        print(f"\tMean squared error: {np.mean(mse)}")
        print(f"\tR2 score: {np.mean(r2)}")
        print("")

        # Find the best seed
        best_seed = np.argmax(r2)
        y_test, y_pred = predicts[best_seed][regressor]
        # print(f"\tBest seed: {best_seed}")

        # Scatter plot of the prediction for the best seed
        plt.title(f'Prediction for {regressor}')
        plt.xlabel('Sample')
        plt.ylabel('Class')
        len_arr = len(y_test)
        plt.scatter(range(len_arr), y_test, label='Real')
        plt.scatter(range(len_arr), y_pred, label='Predicted')
        plt.legend()
        filename = f'./results/prediction_{regressor.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()


if __name__ == '__main__':
    main_classification()
    main_regression()
