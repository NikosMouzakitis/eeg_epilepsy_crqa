import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# testing on one channel 
selected_channel = 'FP1-F7'

def load_single_channel(npy_path, metadata_path, channel_name):
    """Load a single EEG channel into a NumPy array."""
    # Load data and metadata
    data = np.load(npy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    channels = metadata['channels']  # List of channel names
    times = metadata['times']  

    # Find index of the selected channel
    try:
        channel_idx = channels.index(channel_name)
    except ValueError:
        raise ValueError(f"Channel {channel_name} not found in {channels}")

    # Extract the single channel data
    single_channel_data = data[channel_idx]
    
    print(f"Loaded channel {channel_name} with {len(single_channel_data)} samples")
    
    return single_channel_data, times

def plot_single_channel(data, times, channel_name):
    """Plot the single channel data."""
    plt.figure(figsize=(12, 4))
    plt.plot(times, data, linewidth=0.8)
    plt.title(f"EEG Data: Channel {channel_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_phase_space_vectors(segment_data, m=3, tau=1):
    # Check if segment_data is long enough for embedding
    n_samples = len(segment_data)
    if n_samples < m * tau:
        raise ValueError(f"Segment length ({n_samples}) is too short for m={m} and tau={tau}")
    
    # Calculate the number of phase space vectors
    n_vectors = n_samples - (m - 1) * tau
    print("creating {n_vectors} vectors")

    # Initialize array to store phase space vectors
    phase_space_vectors = np.zeros((n_vectors, m))
    
    # Construct phase space vectors using a for loop
    for i in range(n_vectors):
        for j in range(m):
            phase_space_vectors[i, j] = segment_data[i + j * tau]
    
    print(f"Created {n_vectors} phase space vectors with m={m} and tau={tau}")
    
    return phase_space_vectors

def visualize_phase_space(phase_space_vectors):

    if phase_space_vectors.shape[1] != 3:
        raise ValueError("Phase space vectors must have 3 dimensions for 3D visualization")
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = phase_space_vectors[:, 0]
    y = phase_space_vectors[:, 1]
    z = phase_space_vectors[:, 2]
    
    # Plot the trajectory with connected lines in black
    ax.plot(x, y, z, linewidth=1, alpha=0.9, color='blue')
    
    # Set labels
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + τ)')
    ax.set_zlabel('x(t + 2τ)')
    ax.set_title('3D Phase Space Trajectory (m = 3)')
    
    # Set equal axis scaling
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(-1,1) 
    #ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-1,1)
    #ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(-1,1)
    #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def create_recurrence_matrix(phase_space_vectors, radius=0.2):
    """    
            Parameters:
    phase_space_vectors: N x m array of phase space vectors
    radius: recurrence threshold
            Returns:
    recurrence_matrix: N x N binary matrix
    """
    # Calculate pairwise distances
    distances = squareform(pdist(phase_space_vectors, 'euclidean'))
    
    # Create recurrence matrix (1 if distance <= radius, 0 otherwise)
    recurrence_matrix = (distances <= radius).astype(int)
    
    # Remove self-recurrences (main diagonal)
    #keep for now.
    #np.fill_diagonal(recurrence_matrix, 0)
    
    return recurrence_matrix

def calculate_diagonal_lines(recurrence_matrix, min_line_length=2):
    n = recurrence_matrix.shape[0]
    diagonal_lengths = []
    
    # Check all diagonals (offset from main diagonal)
    for k in range(-n + 1, n):
        diag = np.diag(recurrence_matrix, k)
        
        # Find sequences of 1s (recurrences)
        current_length = 0
        for value in diag:
            if value == 1:
                current_length += 1
            else:
                if current_length >= min_line_length:
                    diagonal_lengths.append(current_length)
                current_length = 0
        
        # Check if line continues to the end
        if current_length >= min_line_length:
            diagonal_lengths.append(current_length)
    
    return diagonal_lengths

def calculate_vertical_lines(recurrence_matrix, min_line_length=2):
    n = recurrence_matrix.shape[0]

    vertical_lengths = []
    
    for j in range(n):
        col = recurrence_matrix[:, j]
        
        current_length = 0
        for value in col:
            if value == 1:
                current_length += 1
            else:
                if current_length >= min_line_length:
                    vertical_lengths.append(current_length)
                current_length = 0
        
        if current_length >= min_line_length:
            vertical_lengths.append(current_length)
    
    return vertical_lengths

    #calculation of RQA features
def calculate_rqa_metrics_custom(segment_data, m=3, tau=1, radius=0.1):
    # Create phase space vectors
    phase_space_vectors = create_phase_space_vectors(segment_data, m, tau)
    
    # Create recurrence matrix
    R = create_recurrence_matrix(phase_space_vectors, radius)
    n = R.shape[0]
    total_points = n * n
    
    # Calculate basic statistics
    recurrence_rate = np.sum(R) / total_points
    
    # Diagonal line analysis
    diagonal_lengths = calculate_diagonal_lines(R)
    total_diagonal_lines = len(diagonal_lengths)
    sum_diagonal_lines = sum(diagonal_lengths)
    
    if total_diagonal_lines > 0:
        determinism = sum_diagonal_lines / np.sum(R)
        avg_diagonal_line = sum_diagonal_lines / total_diagonal_lines
        max_diagonal_line = max(diagonal_lengths) if diagonal_lengths else 0
        divergence = 1.0 / max_diagonal_line if max_diagonal_line > 0 else float('inf')
        
        # Entropy of diagonal line lengths
        unique, counts = np.unique(diagonal_lengths, return_counts=True)
        entropy_diagonal = entropy(counts / total_diagonal_lines)
    else:
        determinism = avg_diagonal_line = max_diagonal_line = entropy_diagonal = 0
        divergence = float('inf')
    
    # Vertical line analysis (laminarity)
    vertical_lengths = calculate_vertical_lines(R)
    total_vertical_lines = len(vertical_lengths)
    sum_vertical_lines = sum(vertical_lengths)
    
    if total_vertical_lines > 0:
        laminarity = sum_vertical_lines / np.sum(R)
        avg_vertical_line = sum_vertical_lines / total_vertical_lines
        max_vertical_line = max(vertical_lengths) if vertical_lengths else 0
        trapping_time = avg_vertical_line
    else:
        laminarity = avg_vertical_line = max_vertical_line = trapping_time = 0
    
    # Return all metrics
    metrics = {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'average_diagonal_line': avg_diagonal_line,
        'longest_diagonal_line': max_diagonal_line,
        'divergence': divergence,
        'entropy_diagonal_lines': entropy_diagonal,
        'laminarity': laminarity,
        'trapping_time': trapping_time,
        'longest_vertical_line': max_vertical_line,
        'number_diagonal_lines': total_diagonal_lines,
        'number_vertical_lines': total_vertical_lines,
        'ratio_determ_recurrence': determinism / recurrence_rate if recurrence_rate > 0 else 0
    }
    
    return metrics, R


def plot_rp(recurrence_matrix, title="Recurrence Plot"):
    """
    Plot the custom recurrence matrix.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(recurrence_matrix, cmap='binary', origin='lower', 
               aspect='equal', interpolation='none')
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('Time (samples)')
    plt.colorbar(label='Recurrence')
    plt.tight_layout()
    plt.show()

#   FNN implementation
def false_nearest_neighbors(time_series, tau=1, max_dim=10, rtol=15.0, atol=2.0):
    n = len(time_series)
    fnn_ratio = np.zeros(max_dim)
    
    # For m=1, find nearest neighbors
    phase_space_1 = time_series.reshape(-1, 1)
    distances = squareform(pdist(phase_space_1))
    np.fill_diagonal(distances, np.inf)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_dists = np.min(distances, axis=1)
    
    for m in range(2, max_dim + 1):
        # Create phase space vectors for dimension m
        n_vectors = n - (m-1)*tau
        phase_space_m = np.zeros((n_vectors, m))
        
        for i in range(n_vectors):
            for j in range(m):
                phase_space_m[i, j] = time_series[i + j*tau]
        
        false_count = 0
        valid_pairs = 0
        
        for i in range(n_vectors):
            if i >= len(nearest_indices) or nearest_indices[i] >= n_vectors:
                continue
                
            # Distance in current dimension
            dist_m = np.linalg.norm(phase_space_m[i] - phase_space_m[nearest_indices[i]])
            dist_m_minus_1 = nearest_dists[i]
            
            if dist_m_minus_1 > 1e-10:
                relative_increase = abs(dist_m - dist_m_minus_1) / dist_m_minus_1
                absolute_increase = abs(dist_m - dist_m_minus_1) / np.std(time_series)
                
                if relative_increase > rtol or absolute_increase > atol:
                    false_count += 1
                
                valid_pairs += 1
        
        fnn_ratio[m-1] = false_count / valid_pairs if valid_pairs > 0 else 0
        
        # Update for next dimension
        if m < max_dim:
            distances = squareform(pdist(phase_space_m))
            np.fill_diagonal(distances, np.inf)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_dists = np.min(distances, axis=1)
    
    # Better optimal dimension detection
    # Look for the dimension where FNN drops below 5% and stays low
    for m in range(3, max_dim):
        if fnn_ratio[m] < 0.05 and all(fnn_ratio[m:] < 0.1):
            optimal_dim = m + 1
            break
    else:
        optimal_dim = max_dim  # fallback
    
    return fnn_ratio, optimal_dim



    #Plot of the FNN analysis results.
def plot_fnn_results(fnn_ratio, optimal_dim):
    dimensions = range(1, len(fnn_ratio) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, fnn_ratio, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% threshold')
    plt.axvline(x=optimal_dim, color='g', linestyle='--', alpha=0.7, label=f'Optimal dimension: {optimal_dim}')
    
    plt.xlabel('Embedding Dimension (m)')
    plt.ylabel('Fraction of False Nearest Neighbors')
    plt.title('False Nearest Neighbors Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  #  log scale
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    npy_path = "../../data/p11_06_filtered.npy"
    metadata_path = "../../data/p11_06_metadata.npy"
    SEG_SIZE = 2048
    init_idx =0 
    # Load single channel data on an array.
    channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
    
    # Plot the single channel
    #plot_single_channel(channel_data, times, selected_channel)

    #create a 4-seconds segment.
    segment_data = channel_data[init_idx:init_idx+SEG_SIZE]
    segment_time = times[init_idx:init_idx+SEG_SIZE]
    plot_single_channel(segment_data, segment_time, selected_channel)



    #embedding parameters
    m = 3
    tau = 1
    
    psv = create_phase_space_vectors(segment_data, m = m, tau = tau)

    visualize_phase_space(psv)


    #application of the FNN false nearest neighbors method
    fnn_ratio, optimal_dim = false_nearest_neighbors( segment_data, tau=5, max_dim=10, rtol=15.0, atol=2.0)

    print(f"Optimal embedding dimension: {optimal_dim}")
    print("FNN ratios:", fnn_ratio)
    
    plot_fnn_results(fnn_ratio, optimal_dim)



    # Calculate RQA metrics with custom implementation
    metrics, recurrence_matrix = calculate_rqa_metrics_custom( segment_data, m=optimal_dim, tau=tau, radius=0.1)
    
    print("RQA Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot the recurrence matrix
    plot_rp(recurrence_matrix, f"Recurrence Plot - {selected_channel}")
    

