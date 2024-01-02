# Tools for calculation and visulization

import numpy as np
from scipy.stats import t
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 95% interval
def cal_confidence_interval(sample_data):
    mean_value = np.mean(sample_data)
    std_dev = np.std(sample_data, ddof=1) 

    degrees_of_freedom = len(sample_data) - 1
    alpha = 0.05
    t_critical = t.ppf(1 - alpha/2, degrees_of_freedom)

    margin_of_error = t_critical * (std_dev / np.sqrt(len(sample_data)))
    confidence_interval = (mean_value - margin_of_error, mean_value + margin_of_error)
    
    print(f"Mean: {mean_value}")
    print(f"Margin of error: {margin_of_error}")
    print(f"95% confidence interval: {confidence_interval}")


def t_SNE(features, labels, save_name):
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # Visualize the t-SNE plot
    plt.figure(figsize=(8, 6),dpi=400)
    for label in set(labels.numpy()):
        indices = labels.numpy() == label
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=str(label), alpha=0.7)

    plt.title('t-SNE Visualization')
    plt.savefig(save_name)
    plt.show()


def visulize_para_change(param_changes, path1, path2):
    # Extract layers and change percentages
    layers = list(param_changes.keys())
    change_percentages = list(param_changes.values())

    # Separate layers into weights and biases
    weights = [layer for layer in layers if 'weight' in layer]
    biases = [layer for layer in layers if 'bias' in layer]

    # Plot change percentages for weights
    plt.figure(figsize=(10, 6), dpi=400)
    plt.plot(weights, [param_changes[layer] for layer in weights], marker='o', color='blue', label='Weights')
    plt.xlabel('Layers')
    plt.ylabel('Change Percentage')
    plt.title('Change Percentage for Weights in Each Layer')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(path1)
    plt.show()

    # Plot change percentages for biases
    plt.figure(figsize=(10, 6), dpi=400)
    plt.plot(biases, [param_changes[layer] for layer in biases], marker='o', color='orange', label='Biases')
    plt.xlabel('Layers')
    plt.ylabel('Change Percentage')
    plt.title('Change Percentage for Biases in Each Layer')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(path2)
    plt.show()

