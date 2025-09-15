import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score


# testing on one channel
selected_channel = 'FP1-F7'

def find_optimal_tau_ami(time_series, max_tau=50, bins=16, smooth_window=None):
    """
    Compute AMI (via average_mutual_information) and return the first local minimum
    (τ >= 1). If no local minimum is found, fall back to the 1/e rule.

    Returns: optimal_tau (int), ami_curve (np.array indexed by τ=0..max_tau)
    """
    ami_curve = average_mutual_information(time_series, max_tau=max_tau, bins=bins)

    # optional smoothing (helps with noisy AMI curves)
    if smooth_window and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        ami_curve = uniform_filter1d(ami_curve, size=smooth_window, mode='nearest')

    # robust local-minimum detection (search τ = 1 .. max_tau-1)
    if len(ami_curve) >= 3:
        # check interior points: ami[1]..ami[-2] compared with neighbors
        interior = ami_curve[1:-1]
        left   = ami_curve[:-2]
        right  = ami_curve[2:]
        local_min_mask = (interior < left) & (interior < right)
        local_mins = np.where(local_min_mask)[0]
        if local_mins.size > 0:
            # local_mins[0] corresponds to ami index = (local_mins[0] + 1)
            optimal_tau = int(local_mins[0] + 1)
            return optimal_tau, ami_curve

    # fallback: 1/e rule (first τ where AMI < initial/ e)
    initial_ami = ami_curve[0]
    threshold = initial_ami * (1.0 / np.e)
    candidate_taus = np.where(ami_curve < threshold)[0]
    optimal_tau = int(candidate_taus[0]) if candidate_taus.size > 0 else 1

    return optimal_tau, ami_curve

def average_mutual_information(x, max_tau=50, bins=16):
    """
    Compute Average Mutual Information (AMI) for delays 0...max_tau.

    Parameters:
    x : 1D numpy array (time series)
    max_tau : maximum time delay
    bins : number of bins for histogram

    Returns:
    ami : numpy array of AMI values
    """
    # Digitize signal into bins
    hist, bin_edges = np.histogram(x, bins=bins)
    digitized = np.digitize(x, bin_edges[:-1])

    ami = []
    for tau in range(max_tau+1):
        if tau == 0:
            # AMI at lag 0 is entropy of distribution
            mi = mutual_info_score(digitized, digitized)
        else:
            mi = mutual_info_score(digitized[:-tau], digitized[tau:])
        ami.append(mi)
    return np.array(ami)

def find_optimal_dimension(fnn_ratio, threshold=0.02, stability_window=2):
    """
    Find optimal embedding dimension using the Kennel et al. criterion.
    Returns the first dimension where FNN drops below threshold and stays stable.

    Parameters:
    fnn_ratio: array of FNN percentages for each dimension
    threshold: maximum acceptable FNN percentage (typically 1-5%)
    stability_window: number of consecutive dimensions that must stay low

    Returns:
    optimal_dim: optimal embedding dimension
    """
    # Look for the first dimension where FNN drops below threshold
    # and remains consistently low for the next few dimensions
    for d in range(len(fnn_ratio) - stability_window):
        current_window = fnn_ratio[d:d+stability_window]

        # Check if all values in the window are below threshold
        # and that they don't show an increasing trend (noise amplification)
        if (np.all(current_window <= threshold) and
            not np.any(np.diff(current_window) > 0.005)):  # avoid increasing trend
            return d + 1  # +1 because dimensions start at 1

    # Fallback: return dimension with minimum FNN that's below reasonable threshold
    candidate_dims = np.where(fnn_ratio <= 0.05)[0]  # dimensions with FNN ≤ 5%
    if len(candidate_dims) > 0:
        return candidate_dims[0] + 1

    # Final fallback: dimension with absolute minimum FNN
    return np.argmin(fnn_ratio) + 1

def false_nearest_neighbors(time_series, tau=1, max_dim=10, rtol=15.0, atol=2.0):
    """
    implementινγ the False Nearest Neighbors algorithm based on:
    Kennel, M. B., Brown, R., & Abarbanel, H. D. I. (1992).
    Determining embedding dimension for phase-space reconstruction using a geometrical construction.
    Physical Review A, 45(6), 3403.

    Parameters:
    time_series: 1D array of scalar measurements
    tau: time delay
    max_dim: maximum embedding dimension to test
    rtol: tolerance for relative distance criterion
    atol: tolerance for absolute distance criterion

    Returns:
    fnn_ratio: array of false nearest neighbor ratios for each dimension
    optimal_dim: optimal embedding dimension
    """
    n = len(time_series)
    fnn_ratio = np.zeros(max_dim)
    R_d = np.std(time_series)  # attractor size estimate

    # For each dimension from 1 to max_dim
    for d in range(1, max_dim + 1):
        # Build phase-space vectors in dimension d
        n_vectors = n - (d-1)*tau
        if n_vectors <= 0:
            fnn_ratio[d-1] = 1.0
            continue

        phase_space_d = np.zeros((n_vectors, d))
        for i in range(n_vectors):
            for j in range(d):
                phase_space_d[i, j] = time_series[i + j*tau]

        # Find nearest neighbors in d dimensions using KDTree for efficiency
        tree = KDTree(phase_space_d)
        dists_d, indices_d = tree.query(phase_space_d, k=2)  # k=2 to exclude self
        nearest_dists = dists_d[:, 1]  # distances to nearest neighbor
        nearest_indices = indices_d[:, 1]  # indices of nearest neighbors

        false_count = 0
        valid_pairs = 0

        # Check each point for false neighbors when moving to dimension d+1
        for i in range(n_vectors):
            j = nearest_indices[i]  # index of nearest neighbor in d-dim

            # Skip if we can't compute the (d+1)th coordinate
            if (i + d*tau >= n) or (j + d*tau >= n):
                continue

            dist_d = nearest_dists[i]

            # Get the next coordinate in dimension d+1 (Eq. 3 in paper)
            x_i_next = time_series[i + d*tau]
            x_j_next = time_series[j + d*tau]
            new_coord_diff = x_i_next - x_j_next

            # Compute new distance in d+1 using Pythagorean theorem
            dist_d_plus_1 = np.sqrt(dist_d**2 + new_coord_diff**2)

            # Apply both criteria (Eq. 4 and 5 in paper)
            if dist_d > 1e-10:  # avoid division by zero
                # Criterion 1: Relative distance increase
                criterion1 = (abs(new_coord_diff) / dist_d) > rtol

                # Criterion 2: Absolute distance relative to attractor size
                criterion2 = (dist_d_plus_1 / R_d) > atol

                if criterion1 or criterion2:
                    false_count += 1

                valid_pairs += 1

        fnn_ratio[d-1] = false_count / valid_pairs if valid_pairs > 0 else 0


    optimal_dim = find_optimal_dimension(fnn_ratio, threshold=0.02, stability_window=2)

    return fnn_ratio, optimal_dim

def plot_fnn_results(fnn_ratio, optimal_dim):
    """Plot FNN analysis results with proper formatting"""
    dimensions = range(1, len(fnn_ratio) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, fnn_ratio, 'bo-', linewidth=2, markersize=8, label='FNN ratio')
    plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% threshold')
    plt.axvline(x=optimal_dim, color='g', linestyle='--', alpha=0.7,
                label=f'Optimal dimension: {optimal_dim}')

    plt.xlabel('Embedding Dimension (m)')
    plt.ylabel('Fraction of False Nearest Neighbors')
    plt.title('False Nearest Neighbors Analysis (Corrected Algorithm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def load_single_channel(npy_path, metadata_path, channel_name):
    """Load a single EEG channel into a NumPy array."""
    # Load data and metadata
    data = np.load(npy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    channels = metadata['channels']  # List of channel names
    times = metadata['times']

    # Print shapes for debugging
    print(f"Shape of loaded data: {data.shape}")
    print(f"Expected shape from metadata: ({len(channels)}, {len(times)})")


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
    ax.set_title('3D Phase Space Trajectory')

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
def calculate_rqa_metrics(segment_data, m=3, tau=1, radius=0.1):
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


def plot_ami_results(ami_curve, optimal_tau, max_tau):
    """Plot AMI analysis results with proper formatting"""
    taus = range(len(ami_curve))

    plt.figure(figsize=(10, 6))
    plt.plot(taus, ami_curve, 'bo-', linewidth=2, markersize=8, label='AMI')
    plt.axvline(x=optimal_tau, color='g', linestyle='--', alpha=0.7,
                label=f'Optimal tau: {optimal_tau}')

    plt.xlabel('Time Delay (τ)')
    plt.ylabel('Average Mutual Information')
    plt.title('Average Mutual Information Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    npy_path = "../../data/p11_06_filtered.npy"
    metadata_path = "../../data/p11_06_metadata.npy"
    SEG_SIZE = 1024  # 4 seconds at 256 Hz sampling rate
    
    # Load single channel data
    channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
    
    # Calculate total number of segments
    total_samples = len(channel_data)
    num_segments = total_samples // SEG_SIZE
    print(f"Total samples: {total_samples}, Number of 4-second segments: {num_segments}")
    
    # Lists to store results for each segment
    optimal_taus = []
    optimal_dims = []
    ami_curves = []
    fnn_ratios = []
    segment_indices = []
    
    # Process each segment
    for seg_idx in range(num_segments):
        start_idx = seg_idx * SEG_SIZE
        end_idx = start_idx + SEG_SIZE
        
        # Extract segment
        segment_data = channel_data[start_idx:end_idx]
        segment_time = times[start_idx:end_idx]
        
        print(f"\n--- Processing Segment {seg_idx + 1}/{num_segments} (samples {start_idx}-{end_idx-1}) ---")
        
        # Determine optimal tau using AMI
        max_tau_to_test = 50
        optimal_tau, ami_curve = find_optimal_tau_ami(segment_data, max_tau=max_tau_to_test)
        
        # Apply FNN to find optimal dimension
        fnn_ratio, optimal_dim = false_nearest_neighbors(segment_data, tau=optimal_tau, max_dim=15, rtol=15.0, atol=2.0)
        
        # Store results
        optimal_taus.append(optimal_tau)
        optimal_dims.append(optimal_dim)
        ami_curves.append(ami_curve)
        fnn_ratios.append(fnn_ratio)
        segment_indices.append(seg_idx)
        
        print(f"Segment {seg_idx}: τ = {optimal_tau}, m = {optimal_dim}")
    
    # Plot results across all segments
    plt.figure(figsize=(12, 8))
    
    # Plot optimal tau values
    plt.subplot(2, 1, 1)
    plt.plot(segment_indices, optimal_taus, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Segment Index')
    plt.ylabel('Optimal τ')
    plt.title('Optimal Time Delay (τ) Across Segments')
    plt.grid(True, alpha=0.3)
    
    # Plot optimal dimension values
    plt.subplot(2, 1, 2)
    plt.plot(segment_indices, optimal_dims, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Segment Index')
    plt.ylabel('Optimal Dimension (m)')
    plt.title('Optimal Embedding Dimension (m) Across Segments')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    print("\n=== STATISTICS ACROSS ALL SEGMENTS ===")
    print(f"Average optimal τ: {np.mean(optimal_taus):.2f} ± {np.std(optimal_taus):.2f}")
    print(f"Average optimal m: {np.mean(optimal_dims):.2f} ± {np.std(optimal_dims):.2f}")
    print(f"Most frequent τ: {np.bincount(optimal_taus).argmax()}")
    print(f"Most frequent m: {np.bincount(optimal_dims).argmax()}")
    
    # Plot histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(optimal_taus, bins=range(min(optimal_taus), max(optimal_taus)+2), alpha=0.7, edgecolor='black')
    plt.xlabel('Optimal τ')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal τ Values')
    
    plt.subplot(1, 2, 2)
    plt.hist(optimal_dims, bins=range(min(optimal_dims), max(optimal_dims)+2), alpha=0.7, edgecolor='black')
    plt.xlabel('Optimal Dimension (m)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal m Values')
    
    plt.tight_layout()
    plt.show()
    
    # Select the most representative values
    final_tau = int(np.median(optimal_taus))
    final_dim = int(np.median(optimal_dims))
    
    print(f"\n=== RECOMMENDED VALUES ===")
    print(f"Recommended τ: {final_tau}")
    print(f"Recommended m: {final_dim}")
    
    # Optional: Show detailed results for a few representative segments
    print(f"\n=== DETAILED RESULTS FOR REPRESENTATIVE SEGMENTS ===")
    for i in [0, len(optimal_taus)//2, -1]:  # First, middle, and last segment
        if i < len(optimal_taus):
            print(f"Segment {i}: τ = {optimal_taus[i]}, m = {optimal_dims[i]}")
            # You could add plots for these specific segments if desired

    # Now you can use final_tau and final_dim for your RQA analysis
    print(f"\nUsing recommended values: τ = {final_tau}, m = {final_dim}")
    
    # Example: Process one segment with the recommended values
    segment_idx = 0  # or choose any segment
    segment_data = channel_data[segment_idx*SEG_SIZE:(segment_idx+1)*SEG_SIZE]
    
    # Create phase space vectors with recommended parameters
    psv = create_phase_space_vectors(segment_data, m=final_dim, tau=final_tau)
    
    # Calculate RQA metrics
    data_std = np.std(segment_data)
    epsilon = 0.8 * data_std
    metrics, recurrence_matrix = calculate_rqa_metrics(segment_data, m=final_dim, tau=final_tau, radius=epsilon)
    
    print("\nRQA Metrics with recommended parameters:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
