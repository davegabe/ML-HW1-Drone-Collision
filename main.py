# Launch classification model on different seeds
from matplotlib import pyplot as plt
import numpy as np
from classification_no_angles import main as main_no_angles
from classification import main as main_angles
from multiprocessing import Pool
import builtins

def main():
    """
    Launch classification model on different seeds.
    """
    tot_scores: dict[str, float] = dict()
    best_predict: dict[str, tuple[np.ndarray, np.ndarray]] = dict()

    n_processes = 10

    # Launch the classification model on different seeds
    with Pool(n_processes, initializer=mute) as p:
        results = p.starmap(main_no_angles, [(seed,) for seed in range(n_processes)])
        scores, predicts = zip(*results)

        # Add the scores to the total scores
        for i in range(n_processes):
            for key in scores[i]:
                if key not in tot_scores:
                    tot_scores[key] = []
                # Save the best prediction
                if key not in best_predict or scores[i][key] > max(tot_scores[key]):
                    best_predict[key] = predicts[i][key]
                # Add the score to the total scoress
                tot_scores[key].append(scores[i][key])
        
    
    # Print the average scores
    for key in tot_scores:
        print(f'{key}: {np.mean(tot_scores[key])}')

    # Plot the prediction
    for key in best_predict:
        plt.title(f'Prediction for {key}')
        plt.xlabel('Sample')
        plt.ylabel('Class')
        plt.yticks(np.arange(0, 5, 1))
        len_arr = len(best_predict[key][0])
        plt.plot(range(len_arr), best_predict[key][0], label='Real')
        plt.plot(range(len_arr), best_predict[key][1], label='Predicted')
        plt.legend()
        plt.savefig(f'./results/prediction_{key.replace(" ", "_")}.png', dpi=600)
    

def mute():
    """
    Mute the print function.
    """
    builtins.print = lambda *args, **kwargs: None

if __name__ == '__main__':
    main()