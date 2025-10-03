import numpy as np
from pyrqa.opencl import OpenCL
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy import signal  # Added for filtering
from tqdm.auto import tqdm
from scipy.signal import welch
from scipy.spatial import ConvexHull
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import pyopencl as cl

opencl = OpenCL(platform_id=0,
                device_ids=(0,))
print(opencl)


def save_rp_plot(rp_result, channel1_name, channel2_name, segment_info="", subfolder="rps"):
    """
    Save recurrence plot to images folder with channel names
    
    Parameters:
    rp_result: RP computation result
    channel1_name: name of first channel
    channel2_name: name of second channel
    segment_info: additional info about the segment
    subfolder: subfolder within images
    """
    # Create images directory if it doesn't exist
    base_dir = "images"
    if subfolder:
        save_dir = os.path.join(base_dir, subfolder)
    else:
        save_dir = base_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with channel names
    filename = f"RP_{channel1_name}_vs_{channel2_name}"
    if segment_info:
        filename += f"_{segment_info}"
    
    # Remove any characters that might be problematic in filenames
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    
    filepath = os.path.join(save_dir, f"{filename}.png")
    
    # Create and save the recurrence plot
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(rp_result.recurrence_matrix_reverse, cmap='binary', origin='lower')
    plt.title(f'Recurrence Plot: {channel1_name} vs {channel2_name}')
    plt.xlabel(f'{channel2_name} Samples')
    plt.ylabel(f'{channel1_name} Samples')
    plt.tight_layout()
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    print(f"RP saved to: {filepath}")


def pss(x, m=3, t=1, norm='euclidean'):
    """
    Compute the phase space size (maximal and average diameter) of a time series.

    Parameters:
    x : ndarray
        Input time series (1D array).
    m : int
        Embedding dimension.
    t : int
        Time delay (lag).
    norm : str
        Norm to use ('euclidean', 'maxnorm', 'minnorm').

    Returns:
    max_diameter : float
        Maximal phase space diameter.
    avg_diameter : float
        Average phase space diameter.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim != 1:
        raise ValueError("Input time series must be 1D.")
    if m < 1 or t < 1:
        raise ValueError("Embedding dimension and time delay must be positive integers.")

    # Construct the phase space (delay embedding)
    N = len(x)
    n_vectors = N - (m - 1) * t
    if n_vectors <= 0:
        raise ValueError("Time series too short for given m and t.")

    # Create embedded vectors
    embedded = np.zeros((n_vectors, m))
    for i in range(m):
        embedded[:, i] = x[i * t : i * t + n_vectors]

    # Compute pairwise distances
    if norm == 'euclidean':
        distances = cdist(embedded, embedded, metric='euclidean')
    elif norm == 'maxnorm':
        distances = cdist(embedded, embedded, metric='chebyshev')
    elif norm == 'minnorm':
        distances = cdist(embedded, embedded, metric='cityblock')
    else:
        raise ValueError("Norm must be 'euclidean', 'maxnorm', or 'minnorm'.")

    # Maximal diameter: maximum distance between any two points
    max_diameter = np.max(distances)

    # Average diameter: average of all non-zero distances (excluding self-distances)
    np.fill_diagonal(distances, 0)  # Set diagonal (self-distances) to 0
    non_zero_distances = distances[distances > 0]
    avg_diameter = np.mean(non_zero_distances) if non_zero_distances.size > 0 else 0.0

    return max_diameter, avg_diameter



def find_optimal_tau_ami(time_series, max_tau=50, bins=16, smooth_window=None):
    """
    Compute AMI (via average_mutual_information) and return the first local minimum
    (τ >= 1). If no local minimum is found, fall back to the rule of 1/e.

    Returns: optimal_tau (int), ami_curve (np.array indexed by τ=0..max_tau)
    """
    ami_curve = average_mutual_information(time_series, max_tau=max_tau, bins=bins)

    # optional smoothing (helps with noisy AMI curves)
    if smooth_window and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        ami_curve = uniform_filter1d(ami_curve, size=smooth_window, mode='nearest')

    #  local-minimum detection (search τ = 1 .. max_tau-1)
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
    Computation of Average Mutual Information (AMI) for selected delays 0...max_tau.

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

            # Get the next coordinate in dimension d+1
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


def visualize_phase_space(phase_space_vectors, title, ax=None):
    if phase_space_vectors.shape[1] != 3:
        raise ValueError("Phase space vectors must have 3 dimensions for 3D visualization")

    # If no ax is provided, create a new figure with 3D axes
    if ax is None:
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
    ax.set_title(f'{title} - Trajectory')

    # Set equal axis scaling
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def compare_phase_space(normal_psv, epileptic_psv):
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(16, 6))  # Adjusted figure size for side-by-side plots
    fig.suptitle('Phase Space Trajectories Comparison', fontsize=16, y=1.05)
    # Create first subplot for normal phase space
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_phase_space(normal_psv[:, :3], "Normal", ax=ax1)

    # Create second subplot for epileptic phase space
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_phase_space(epileptic_psv[:, :3], "Epileptic", ax=ax2)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

print("hi")



npy_path = "p24_01_filtered.npy"
metadata_path = "p24_01_metadata.npy"

    # Load sampling frequency from metadata
metadata = np.load(metadata_path, allow_pickle=True).item()
sfreq = metadata['sfreq']  # Sampling frequency in Hz
print(f"Sampling frequency: {sfreq} Hz")
"""
    File Name: chb24_01.edf
    Number of Seizures in File: 2
    Seizure Start Time: 480 seconds    sample 122800
    Seizure End Time: 505 seconds     sample  129280
    Seizure Start Time: 2451 seconds  sample 627456
    Seizure End Time: 2476 seconds    sample 633856
"""
SEG_SIZE = 256
#set epileptic sample index and normal sample index
#epileptic_idx = 125456
epileptic_idx = 629000

normal_idx = 0
normal_idx = 100050

    # Load single channel data on an array.


# Load metadata
metadata = np.load(metadata_path, allow_pickle=True).item()
sfreq = metadata['sfreq']  # Sampling frequency in Hz
print(f"Sampling frequency: {sfreq} Hz")

# Function to load single channel data (assumed to be defined)
def load_single_channel(npy_path, metadata_path, selected_channel):
    data = np.load(npy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    sfreq = metadata['sfreq']
    times = np.arange(data.shape[1]) / sfreq
    return data[selected_channel], times


from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
import multiprocessing as mp

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation, RPComputation
import os

def compute_cross_rqa(segment1, segment2, embedding_dimension=3, time_delay=1, channel1_name="Channel1", channel2_name="Channel2", segment_idx=0, save_rp=False):

    print(f"compute crqa in process {mp.current_process().name}")
    min_length = min(len(segment1), len(segment2))
    segment1 = segment1[:min_length]
    segment2 = segment2[:min_length]
    radius_fraction = 0.1
    norm = 'euclidean'
    max_d1, avg_d1 = pss(segment1, m=embedding_dimension, t=time_delay, norm=norm)
    max_d2, avg_d2 = pss(segment2, m=embedding_dimension, t=time_delay, norm=norm)
    mean_diameter = (max_d1 + max_d2) / 2
    radius = mean_diameter * radius_fraction
    print(f"Computed radius: {radius} (fraction {radius_fraction} of mean diameter {mean_diameter})")
    ts1 = TimeSeries(segment1, embedding_dimension=embedding_dimension, time_delay=time_delay)
    ts2 = TimeSeries(segment2, embedding_dimension=embedding_dimension, time_delay=time_delay)
    settings = Settings(
        [ts1, ts2],
        analysis_type=Cross,
        neighbourhood=FixedRadius(radius),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )
    print(f"running RQA metrics computation m={embedding_dimension} tau={time_delay}")

    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()

     # Calculate and save RP plot
     #hardcode no.
    save_rp=False
    if save_rp:
        print("calculating and saving RP plot")
        rp_computation = RPComputation.create(settings)
        rp_result = rp_computation.run()

        # Save the recurrence plot
        save_rp_plot(rp_result, channel1_name, channel2_name, f"seg{segment_idx}")

    return result

# Worker function for each process
def compute_rqa_for_electrode1(args):
    electrode1, windows_by_channel, labels_by_channel, total_windows, selected_channels, channel_names = args
    print(f"Starting process for electrode1={electrode1}")
    
    # Initialize partial RQA matrix for this electrode1
    partial_rqa_matrix = np.zeros((total_windows, len(selected_channels), 17))
    
    for window_idx in range(total_windows):
        for electrode2 in range(len(selected_channels)):
            window1 = windows_by_channel[electrode1][window_idx]
            window2 = windows_by_channel[electrode2][window_idx]
            
            # Get segment info for naming
            seg_idx, win_idx_within_seg, start_sample, end_sample = window_info_by_channel[0][window_idx]
            
            print(f"Computing Cross RQA for window {window_idx+1}/{total_windows} "
                  f"(Segment {seg_idx}, Window {win_idx_within_seg}) "
                  f"(Channel {electrode1} vs Channel {electrode2})")
            
            # Find optimal parameters
            max_tau_to_test = 25
            s1_otau, _ = find_optimal_tau_ami(window1, max_tau=max_tau_to_test)
            s2_otau, _ = find_optimal_tau_ami(window2, max_tau=max_tau_to_test)
            optimal_tau = min(s1_otau, s2_otau)
            
            fnn_ratio, s1_odim = false_nearest_neighbors(window1, tau=optimal_tau, max_dim=10, rtol=15.0, atol=2.0)
            fnn_ratio, s2_odim = false_nearest_neighbors(window2, tau=optimal_tau, max_dim=10, rtol=15.0, atol=2.0)
            optimal_dim = min(s1_odim, s2_odim)
            
            print(f"optimal_tau: {optimal_tau}, optimal_dim: {optimal_dim}")
            
            # Get channel names
            ch1_name = channel_names[electrode1] if channel_names else f"Ch{electrode1}"
            ch2_name = channel_names[electrode2] if channel_names else f"Ch{electrode2}"
            
            result = compute_cross_rqa( window1, window2, embedding_dimension=optimal_dim, time_delay=optimal_tau, channel1_name=ch1_name, channel2_name=ch2_name, segment_idx=f"seg{seg_idx}_win{win_idx_within_seg}", save_rp=True )
            
            features = np.array([
                result.recurrence_rate if result.recurrence_rate > 0 else 0.0,
                result.determinism if not np.isnan(result.determinism) else 0.0,
                result.average_diagonal_line if not np.isnan(result.average_diagonal_line) else 0.0,
                result.longest_diagonal_line,
                result.divergence if not np.isinf(result.divergence) else 0.0,
                result.entropy_diagonal_lines if not np.isnan(result.entropy_diagonal_lines) else 0.0,
                result.laminarity if not np.isnan(result.laminarity) else 0.0,
                result.trapping_time if not np.isnan(result.trapping_time) else 0.0,
                result.longest_vertical_line,
                result.average_white_vertical_line if not np.isnan(result.average_white_vertical_line) else 0.0,
                result.longest_white_vertical_line,
                result.longest_white_vertical_line_inverse if not np.isnan(result.longest_white_vertical_line_inverse) else 0.0,
                result.entropy_vertical_lines if not np.isnan(result.entropy_vertical_lines) else 0.0,
                result.entropy_white_vertical_lines if not np.isnan(result.entropy_white_vertical_lines) else 0.0,
                result.ratio_determinism_recurrence_rate if not np.isnan(result.ratio_determinism_recurrence_rate) else 0.0,
                result.ratio_laminarity_determinism if not np.isnan(result.ratio_laminarity_determinism) and not np.isinf(result.ratio_laminarity_determinism) else 0.0,
                labels_by_channel[electrode1][window_idx]  # Label for this window
            ])
            
            print(f"Features for window {window_idx+1}, ch{electrode1}_ch{electrode2}: {features[:5]}...")  # Show first 5
            partial_rqa_matrix[window_idx, electrode2, :] = features
    
    return electrode1, partial_rqa_matrix

def extract_non_overlapping_windows(channel_data, segment_boundaries, window_size=256):
    """
    Extract non-overlapping windows from each segment boundary.
    
    Parameters:
    channel_data: Full channel data array
    segment_boundaries: List of segment boundaries with labels
    window_size: Size of each window (default: 256)
    
    Returns:
    all_windows: List of window arrays
    all_labels: List of labels for each window
    window_info: List of tuples (segment_idx, window_idx, start_sample, end_sample)
    """
    all_windows = []
    all_labels = []
    window_info = []
    
    for seg_idx, boundary in enumerate(segment_boundaries):
        start_sample = boundary['start_sample']
        end_sample = boundary['end_sample'] if boundary['end_sample'] is not None else len(channel_data)
        label = boundary['label']
        
        # Ensure the segment is within data bounds
        if start_sample < len(channel_data) and end_sample <= len(channel_data) and start_sample < end_sample:
            segment_length = end_sample - start_sample
            
            # Calculate how many complete windows we can extract
            num_windows = segment_length // window_size
            
            # Extract non-overlapping windows
            for window_idx in range(num_windows):
                window_start = start_sample + window_idx * window_size
                window_end = window_start + window_size
                window_data = channel_data[window_start:window_end]
                
                all_windows.append(window_data)
                all_labels.append(label)
                window_info.append((seg_idx, window_idx, window_start, window_end))
                
            print(f"Segment {seg_idx} ({'Epileptic' if label == 1 else 'Normal'}): "
                  f"extracted {num_windows} windows from {segment_length} samples "
                  f"(remaining: {segment_length % window_size} samples)")
        else:
            print(f"Invalid segment {seg_idx}: samples {start_sample} to {end_sample}")
    
    return all_windows, all_labels, window_info


# Main execution

# Your segment boundaries
segment_boundaries = [
#    {'start_sample': 0, 'end_sample': 122800, 'label': 0},  # Normal: before first seizure
    {'start_sample': 122800, 'end_sample': 129280, 'label': 1},  # Epileptic: first seizure
#    {'start_sample': 129280, 'end_sample': 627456, 'label': 0},  # Normal: between seizures
    {'start_sample': 627456, 'end_sample': 633856, 'label': 1},  # Epileptic: second seizure
#    {'start_sample': 633856, 'end_sample': None, 'label': 0}  # Normal: after second seizure
]

SEL_SIZE = 256

# Load metadata
metadata = np.load(metadata_path, allow_pickle=True).item()
sfreq = metadata['sfreq']
print(f"Sampling frequency: {sfreq} Hz")


channel_names = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',  'FP1-F3',   'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',  'FP2-F8',   'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ',  'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'

]


# Process channels
selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#selected_channels = [0, 1]
windows_by_channel = []
labels_by_channel = []
window_info_by_channel = []

for ch_idx in selected_channels:
    print(f"\nProcessing channel {ch_idx}")
    channel_data, times = load_single_channel(npy_path, metadata_path, ch_idx)
    print(f"Channel {ch_idx} - Loaded channel data with shape: {channel_data.shape}")
    
    # Extract non-overlapping windows
    windows, labels, window_info = extract_non_overlapping_windows(
        channel_data, segment_boundaries, window_size=SEL_SIZE
    )
    
    windows_by_channel.append(windows)
    labels_by_channel.append(labels)
    window_info_by_channel.append(window_info)
    
    print(f"Channel {ch_idx} - Total windows extracted: {len(windows)}")
    print(f"Channel {ch_idx} - Window lengths: {[len(win) for win in windows[:5]]}...")  # Show first 5

# Get total number of windows
total_windows = len(windows_by_channel[0])
print(f"\nTotal windows to process: {total_windows}")
print(f"Total channels: {len(selected_channels)}")

# Initialize RQA matrix: [windows, electrode1, electrode2, 17_features]
rqa_matrix = np.zeros((total_windows, len(selected_channels), len(selected_channels), 17))

# Prepare arguments for each process (now with channel_names)
args_list = [
    (electrode1, windows_by_channel, labels_by_channel, total_windows, selected_channels, channel_names)
    for electrode1 in range(len(selected_channels))
]

# Create a process pool
num_processes = min(len(selected_channels), cpu_count())
print(f"Using {num_processes} processes for parallel computation")

with ThreadPool(processes=num_processes) as pool:
    # Map the worker function to the arguments
    results = pool.map(compute_rqa_for_electrode1, args_list)

    # Collect results into rqa_matrix
    for electrode1, partial_rqa_matrix in results:
        rqa_matrix[:, electrode1, :, :] = partial_rqa_matrix

    print("RQA computation completed")
    print(f"Shape of rqa_matrix: {rqa_matrix.shape}")

print("RQA computation completed")
print(f"Final shape of rqa_matrix: {rqa_matrix.shape}")

# Save the matrix
np.save('rqa_matrix_all_windows.npy', rqa_matrix)
print("rqa_matrix saved to 'rqa_matrix_all_windows.npy'")

print(rqa_matrix)


# Also save window info for reference
window_data = {
    'window_info': window_info_by_channel[0],  # Same for all channels
    'labels': labels_by_channel[0],  # Same for all channels
    'channel_names': channel_names,
    'selected_channels': selected_channels
}
np.save('window_info.npy', window_data)
print("Window info saved to 'window_info.npy'")
