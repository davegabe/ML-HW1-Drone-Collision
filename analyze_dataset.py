import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_distribution(y):
    """
    Plots the distribution of the dataset.
    """
    # Plot the distribution of the dataset into the 5 classes (0, 1, 2, 3, 4) showing the percentage of each class
    print(" - Plotting the distribution of the dataset...")
    sns.countplot(x=y)
    plt.title("Distribution of the dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.yticks(np.arange(0, 600, 50))
    # Over the bars, show the percentage of each class
    for p in plt.gca().patches:
        plt.gca().text(p.get_x() + p.get_width()/2., p.get_height(), '{:1.2f}%'.format(100*p.get_height()/len(y)), 
                fontsize=10, color='black', ha='center', va='bottom')
    
    # Save the plot
    plt.savefig("./results/distribution.png", dpi=600)

def main():
    """
    Plots the distribution of the dataset.
    """
    # Load the dataset
    print(" - Loading the dataset...")
    dataset = pd.read_csv("./data/train_set.tsv", sep='\t', header=0)

    # Split the dataset into features and labels
    X = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -2]

    # Plot the distribution of the dataset
    plot_distribution(y)


if __name__ == '__main__':
    main()