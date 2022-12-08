# Launch classification model on different seeds
from matplotlib import pyplot as plt
import numpy as np
from classification_no_angles import main as main_no_angles
from classification import main as main_angles
from multiprocessing import Pool
import builtins
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import seaborn as sns

def main():
    """
    Launch classification model on different seeds.
    """
    # List of predictions for each seed, each prediction is a dictionary with the name of the classifier as key and (y_test, y_pred) as value
    predicts: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    n_processes = 4

    # Launch the classification model on different seeds
    with Pool(n_processes, initializer=mute) as p:
        predicts = p.starmap(main_no_angles, [(seed,) for seed in range(n_processes)])

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
            precision.append(precision_score(y_test, y_pred, average="weighted"))
            recall.append(recall_score(y_test, y_pred, average="weighted"))
            f1.append(f1_score(y_test, y_pred, average="weighted"))

        print(f"{classifier}:")
        print(f"\tAccuracy: {np.mean(accuracy)}")
        print(f"\tPrecision: {np.mean(precision)}")
        print(f"\tRecall: {np.mean(recall)}")
        print(f"\tF1-score: {np.mean(f1)}")

        # Find the best seed
        best_seed = np.argmax(f1)
        print(f"\tBest seed: {best_seed}")
        y_test, y_pred = predicts[best_seed][classifier]

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
        plt.savefig(f'./results/prediction_{classifier.replace(" ", "_")}.png', dpi=600)
        plt.close()
    

def mute():
    """
    Mute the print function.
    """
    builtins.print = lambda *args, **kwargs: None

if __name__ == '__main__':
    main()