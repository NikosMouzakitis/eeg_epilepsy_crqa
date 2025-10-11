import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# 1. Load the matrices
# -----------------------------
file1 = "p24_04_epileptic512.npy"  
file2="p24_1_epileptic_512.npy"                                                                                                
file3 = "p24_3_1_normal512.npy"                                                                                                  
file4 = "p24_3_3_normal512.npy"
file5= "p24_3_epileptic_256.npy"
file6 = "p24_1_3_normal_512.npy"
file7 = "p24_04_1_normal512.npy"
file8="p24_06_epileptic512.npy"
file9="p24_06_normal512.npy"
rqa1 = np.load(file1)  # shape: [samples1, 22, 22, 17]
rqa2 = np.load(file2)  # shape: [samples2, 22, 22, 17]
rqa3 = np.load(file3)  # shape: [samples2, 22, 22, 17]
rqa4 = np.load(file4)  # shape: [samples2, 22, 22, 17]
rqa5 = np.load(file5)  # shape: [samples2, 22, 22, 17]
rqa6 = np.load(file6)  # shape: [samples2, 22, 22, 17]
rqa7 = np.load(file7)  # shape: [samples2, 22, 22, 17]
rqa8 = np.load(file8)  # shape: [samples2, 22, 22, 17]
rqa9 = np.load(file9)  # shape: [samples2, 22, 22, 17]
print("------------------------------_")
print(rqa1.shape)
print(rqa2.shape)
print(rqa3.shape)
print(rqa4.shape)
print(rqa5.shape)
print(rqa6.shape)
print(rqa7.shape)
print(rqa8.shape)
print(rqa9.shape)

# -----------------------------
# 2. Concatenate along first dimension
# -----------------------------

rqa_all = np.concatenate([rqa1, rqa2,rqa3,rqa4,rqa5,rqa6,rqa7,rqa8,rqa9], axis=0)  # shape: [samples1+samples2, 22, 22, 17]


print("------------------------------_")
print("total shape")
print(rqa_all.shape)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def compute_fc(rqa_all, feature_idx=0, plot=True, feature_name=None, threshold=None):
    """
    Computes average functional connectivity matrices for epileptic vs normal,
    using a selected RQA feature. Optionally applies a threshold and plots heatmaps
    and circular network graphs. Connections below the threshold are set to zero.
    
    Args:
        rqa_all: np.array, shape [num_windows, channels, channels, features+1 label]
        feature_idx: int, index of the feature to use (0-based)
        plot: bool, whether to plot heatmaps and network graphs
        feature_name: str, optional name for title
        threshold: float, optional threshold to zero out weak connections
    Returns:
        fc_epileptic, fc_normal: np.arrays of shape [channels, channels]
    """
    # Extract feature matrices
    feature_matrices = rqa_all[..., feature_idx]   # shape: [num_windows, 22, 22]
    
    # Extract labels (last feature in each sample)
    labels = rqa_all[:, 0, 0, -1].astype(int)     # shape: [num_windows]

    # Separate by label
    epileptic_matrices = feature_matrices[labels == 1]
    normal_matrices = feature_matrices[labels == 0]

    # Average across windows
    fc_epileptic = epileptic_matrices.mean(axis=0)
    fc_normal = normal_matrices.mean(axis=0)

    # Apply threshold if given
    if threshold is not None:
        fc_epileptic = np.where(fc_epileptic >= threshold, fc_epileptic, 0)
        fc_normal = np.where(fc_normal >= threshold, fc_normal, 0)

    if plot:
        fname = f"Feature {feature_idx+1}" if feature_name is None else feature_name

        # ------------------------------
        # 1. Heatmaps
        # ------------------------------
        vmin = min(fc_normal.min(), fc_epileptic.min())
        vmax = max(fc_normal.max(), fc_epileptic.max())

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        sns.heatmap(fc_normal, cmap="coolwarm", annot=True, fmt=".2f", vmin=vmin, vmax=vmax)
        plt.title(f"Functional Connectivity - Normal ({fname})")
        plt.xlabel("Channels")
        plt.ylabel("Channels")

        plt.subplot(1,2,2)
        sns.heatmap(fc_epileptic, cmap="coolwarm", annot=True, fmt=".2f", vmin=vmin, vmax=vmax)
        plt.title(f"Functional Connectivity - Epileptic ({fname})")
        plt.xlabel("Channels")
        plt.ylabel("Channels")

        plt.tight_layout()
        plt.show()

        # ------------------------------
        # 2. Circular network graph
        # ------------------------------
        def plot_circular_graph(fc_matrix, title="FC Circular Graph"):
            G = nx.Graph()
            num_channels = fc_matrix.shape[0]
            nodes = list(range(num_channels))
            G.add_nodes_from(nodes)

            # Add edges (already thresholded)
            for i in range(num_channels):
                for j in range(i+1, num_channels):
                    weight = fc_matrix[i, j]
                    if weight > 0:  # only show connections above threshold
                        G.add_edge(i, j, weight=weight)

            # Circular layout
            pos = nx.circular_layout(G)
            edges = G.edges()
            weights = [G[u][v]['weight']*5 for u,v in edges]  # scale for plotting

            plt.figure(figsize=(8,8))
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800,
                    edge_color='orange', width=weights)
            plt.title(title)
            plt.show()

        plot_circular_graph(fc_normal, title=f"Normal - {fname} FC")
        plot_circular_graph(fc_epileptic, title=f"Epileptic - {fname} FC")

    return fc_epileptic, fc_normal


# -----------------------------
# 3. Example usage
# -----------------------------
# Select feature index (0-based)
feature_idx = 1  # Feature 1

# Compute FC matrices and plot
fc_epileptic, fc_normal = compute_fc(rqa_all, feature_idx=feature_idx,plot=True,feature_name="DET",threshold=0.83)

