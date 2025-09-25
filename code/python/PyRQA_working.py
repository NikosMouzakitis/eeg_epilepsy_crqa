import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from google.colab import drive, files
from scipy import signal  # Added for filtering
from tqdm.auto import tqdm
from scipy.signal import welch
from scipy.spatial import ConvexHull
from numpy.linalg import norm

# testing on one channel
selected_channel = 'P8-O2'
selected_channel = 'FT10-T8'
selected_channel = 'FP1-F7'

# Mount Google Drive
drive.mount('/content/drive')


def filter_signal(data, sfreq, low_freq, high_freq, order=4):
    """
    Apply a bandpass filter to isolate a specific frequency band.

    Parameters:
    data: 1D numpy array (time series)
    sfreq: Sampling frequency (Hz)
    low_freq: Lower frequency bound (Hz)
    high_freq: Upper frequency bound (Hz)
    order: Butterworth filter order
    Returns:
    filtered_data: Filtered time series
    """
    nyquist = sfreq / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


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
    plt.title('False Nearest Neighbors Analysis')
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








npy_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_filtered.npy"
metadata_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_metadata.npy"
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
SEG_SIZE = 768
SEG_SIZE = 512
#set epileptic sample index and normal sample index
#epileptic_idx = 125456
epileptic_idx = 629000

normal_idx = 0
normal_idx = 100050

    # Load single channel data on an array.
channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
print(f"Loaded channel data with shape: {channel_data.shape}")
print(f"Loaded times with shape: {times.shape}")
print(f"Loaded metadata with keys: {metadata.keys()}")

# Plot the single channel
plot_single_channel(channel_data, times, selected_channel)







import numpy as np
import matplotlib.pyplot as plt

# File paths
npy_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_filtered.npy"
metadata_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_metadata.npy"

# Load metadata
metadata = np.load(metadata_path, allow_pickle=True).item()
sfreq = metadata['sfreq']  # Sampling frequency in Hz
print(f"Sampling frequency: {sfreq} Hz")

# Define segment boundaries based on provided description
segment_boundaries = [
    {'start_sample': 0, 'end_sample': 122800, 'label': 0},  # Normal: before first seizure
    {'start_sample': 122800, 'end_sample': 129280, 'label': 1},  # Epileptic: first seizure
    {'start_sample': 129280, 'end_sample': 627456, 'label': 0},  # Normal: between seizures
    {'start_sample': 627456, 'end_sample': 633856, 'label': 1},  # Epileptic: second seizure
    {'start_sample': 633856, 'end_sample': None, 'label': 0}  # Normal: after second seizure
]

# Function to load single channel data (assumed to be defined)
def load_single_channel(npy_path, metadata_path, selected_channel):
    data = np.load(npy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    sfreq = metadata['sfreq']
    times = np.arange(data.shape[1]) / sfreq
    return data[selected_channel], times

# Function to extract continuous segments based on boundaries
def extract_continuous_segments(channel_data, segment_boundaries, sfreq):
    segments = []
    labels = []
    time_axes = []

    for boundary in segment_boundaries:
        start_sample = boundary['start_sample']
        end_sample = boundary['end_sample'] if boundary['end_sample'] is not None else len(channel_data)
        label = boundary['label']

        # Ensure the segment is within data bounds
        if start_sample < len(channel_data) and end_sample <= len(channel_data) and start_sample < end_sample:
            segment = channel_data[start_sample:end_sample]
            segments.append(segment)
            labels.append(label)
            # Create time axis for the segment
            time_axis = np.arange(end_sample - start_sample) / sfreq
            time_axes.append(time_axis)
            print(f"Extracted {'Epileptic' if label == 1 else 'Normal'} segment: "
                  f"samples {start_sample} to {end_sample}, length {end_sample - start_sample}")
        else:
            print(f"Invalid segment: samples {start_sample} to {end_sample} out of bounds or invalid.")

    return segments, labels, time_axes

# Load data
selected_channel = 21  # Adjust as needed
channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
print(f"Loaded channel data with shape: {channel_data.shape}")
print(f"Loaded times with shape: {times.shape}")
print(f"Loaded metadata with keys: {metadata.keys()}")

# Extract segments
segments, labels, time_axes = extract_continuous_segments(channel_data, segment_boundaries, sfreq)

# Print segment information
print(f"Total segments extracted: {len(segments)}")
print(f"Segment lengths: {[len(seg) for seg in segments]}")
print(f"Labels: {labels}")

# Plot segments
def plot_segments(segments, labels, time_axes, sfreq):
    for i, (segment, label, time_axis) in enumerate(zip(segments, labels, time_axes)):
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, segment)
        plt.title(f"Segment {i+1} - {'Epileptic' if label == 1 else 'Normal'} "
                  f"(samples {segment_boundaries[i]['start_sample']} to "
                  f"{segment_boundaries[i]['end_sample'] or len(channel_data)})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

plot_segments(segments, labels, time_axes, sfreq)






import numpy as np
import matplotlib.pyplot as plt
!pip install pyrqa

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation, RPComputation
import os

# File paths
npy_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_filtered.npy"
metadata_path = "/content/drive/My Drive/EEG_Processed/pat24/epileptic/p24_01_metadata.npy"

# Load metadata
metadata = np.load(metadata_path, allow_pickle=True).item()
sfreq = metadata['sfreq']  # Sampling frequency in Hz
print(f"Sampling frequency: {sfreq} Hz")

# Define segment boundaries - original
segment_boundaries = [
    {'start_sample': 0, 'end_sample': 122800, 'label': 0},  # Normal
    {'start_sample': 122800, 'end_sample': 129280, 'label': 1},  # Epileptic
    {'start_sample': 129280, 'end_sample': 627456, 'label': 0},  # Normal
    {'start_sample': 627456, 'end_sample': 633856, 'label': 1},  # Epileptic
    {'start_sample': 633856, 'end_sample': None, 'label': 0}  # Normal
]

# Define segment boundaries - original
segment_boundaries = [
    {'start_sample': 0, 'end_sample': 2048, 'label': 0},  # Normal
    {'start_sample': 126800, 'end_sample': 128848, 'label': 1}  # Epileptic
]

# Function to load single channel data (provided, unchanged)
def load_single_channel(npy_path, metadata_path, selected_channel):
    data = np.load(npy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    sfreq = metadata['sfreq']
    times = np.arange(data.shape[1]) / sfreq
    return data[selected_channel], times

# Function to extract continuous segments
def extract_continuous_segments(channel_data, segment_boundaries, sfreq):
    segments = []
    labels = []
    time_axes = []

    for boundary in segment_boundaries:
        start_sample = boundary['start_sample']
        end_sample = boundary['end_sample'] if boundary['end_sample'] is not None else len(channel_data)
        label = boundary['label']

        if start_sample < len(channel_data) and end_sample <= len(channel_data) and start_sample < end_sample:
            segment = channel_data[start_sample:end_sample]
            segments.append(segment)
            labels.append(label)
            time_axis = np.arange(end_sample - start_sample) / sfreq
            time_axes.append(time_axis)
            print(f"Extracted {'Epileptic' if label == 1 else 'Normal'} segment: "
                  f"samples {start_sample} to {end_sample}, length {end_sample - start_sample}")
        else:
            print(f"Invalid segment: samples {start_sample} to {end_sample} out of bounds or invalid.")

    return segments, labels, time_axes

# Function to compute Cross RQA using PyRQA
def compute_cross_rqa(segment1, segment2, embedding_dimension=3, time_delay=1, radius=0.1):
    # Truncate segments to the same length
    min_length = min(len(segment1), len(segment2))
    segment1 = segment1[:min_length]
    segment2 = segment2[:min_length]

    # Create TimeSeries objects
    ts1 = TimeSeries(segment1, embedding_dimension=embedding_dimension, time_delay=time_delay)
    ts2 = TimeSeries(segment2, embedding_dimension=embedding_dimension, time_delay=time_delay)

    # Set up Cross RQA settings
    settings = Settings(
        [ts1, ts2],
        analysis_type=Cross,
        neighbourhood=FixedRadius(radius),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )

    # Run RQA computation
    print("running RQA metrics computation")
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()

    print("calculating crp for plot")
    # Run RP computation for the recurrence plot
    rp_computation = RPComputation.create(settings)
    rp_result = rp_computation.run()

    # Visualize the Cross Recurrence Plot inline
    plt.figure(figsize=(8, 8))
    plt.imshow(rp_result.recurrence_matrix_reverse, cmap='binary', origin='lower')
    plt.title("Cross Recurrence Plot: Segment")
    plt.xlabel(f"Channel Time")
    plt.ylabel(f"Channel Time")
    plt.colorbar(label='Recurrence')
    plt.show()


    return result

# Function to plot segments (optional)
def plot_segments(segments, labels, time_axes, channel_idx, sfreq):
    for i, (segment, label, time_axis) in enumerate(zip(segments, labels, time_axes)):
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, segment)
        plt.title(f"Channel {channel_idx} - Segment {i+1} - {'Epileptic' if label == 1 else 'Normal'} "
                  f"(samples {segment_boundaries[i]['start_sample']} to "
                  f"{segment_boundaries[i]['end_sample'] or len(segment)})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()



########### execution #############

# Process channels separately
selected_channels = [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

#tested with only 3 first channels .. took like an hour+ in Colab.
#selected_channels = [0, 1, 2]

segments_by_channel = []
labels_by_channel = []
time_axes_by_channel = []

for ch_idx in selected_channels:
    print(f"\nProcessing channel {ch_idx}")
    # Load data for one channel at a time
    channel_data, times = load_single_channel(npy_path, metadata_path, ch_idx)
    print(f"Channel {ch_idx} - Loaded channel data with shape: {channel_data.shape}")
    print(f"Channel {ch_idx} - Loaded times with shape: {times.shape}")

    # Extract segments
    segments, labels, time_axes = extract_continuous_segments(channel_data, segment_boundaries, sfreq)
    segments_by_channel.append(segments)
    labels_by_channel.append(labels)
    time_axes_by_channel.append(time_axes)

    # Print segment information
    print(f"Channel {ch_idx} - Total segments extracted: {len(segments)}")
    print(f"Channel {ch_idx} - Segment lengths: {[len(seg) for seg in segments]}")
    print(f"Channel {ch_idx} - Labels: {labels}")

    # Optional: Plot segments
    #plot_segments(segments, labels, time_axes, ch_idx, sfreq)

# Compute Cross RQA for corresponding segment pairs

rqa_results = []

N = len(segment_boundaries)
rqa_matrix = np.zeros((N, len(selected_channels), len(selected_channels), 17))  # Shape: (segments, electrodes, electrodes, 16 features + label)


for i in range(len(segment_boundaries)):
    for electrode1 in range(len(selected_channels)):
        for electrode2 in range(len(selected_channels)):
            segment1 = segments_by_channel[electrode1][i]  # Segment i from channel 0
            segment2 = segments_by_channel[electrode2][i]  # Segment i from channel 1
            print(f"\nComputing Cross RQA for segment {i+1} (Channel {electrode1} vs Channel {electrode2})")

            ## find tau for both segments
            max_tau_to_test = 25
            s1_otau, ami_curve = find_optimal_tau_ami(segment1, max_tau=max_tau_to_test)
            s2_otau, ami_curve = find_optimal_tau_ami(segment2, max_tau=max_tau_to_test)
                ##minimum tau as optimal tau
            optimal_tau = min(s1_otau, s2_otau)
            #optimal_tau = 4
            ## find m   for both segments
            fnn_ratio, s1_odim = false_nearest_neighbors( segment1, tau=optimal_tau, max_dim=15, rtol=15.0, atol=2.0)
            fnn_ratio, s2_odim = false_nearest_neighbors( segment2, tau=optimal_tau, max_dim=15, rtol=15.0, atol=2.0)
            optimal_dim = min(s1_odim, s2_odim)
            #optimal_dim=5
            ##minimum tau as optimal dimension
            print(f"optimal_tau: {optimal_tau}")
            print(f"optimal_dim: {optimal_dim}")

            # Compute Cross RQA
            result = compute_cross_rqa(
                segment1,
                segment2,
                embedding_dimension=optimal_dim,  # Adjust based on analysis (e.g., from FNN)
                time_delay=optimal_tau,          # Adjust based on analysis (e.g., from AMI)
                radius=0.1             # Adjust based on data scale (e.g., std dev)
            )

            print("Attributes of RQAResult:", dir(result))

           # Extract 13 features (adjust based on actual PyRQA output)
           # Extract 16 features
            features = np.array([
                    result.recurrence_rate,
                    result.determinism,
                    result.average_diagonal_line,
                    result.longest_diagonal_line,
                    result.divergence,
                    result.entropy_diagonal_lines,
                    result.laminarity,
                    result.trapping_time,
                    result.longest_vertical_line,
                    result.average_white_vertical_line,
                    result.longest_white_vertical_line,
                    result.longest_white_vertical_line_inverse,
                    result.entropy_vertical_lines,
                    result.entropy_white_vertical_lines,
                    result.ratio_determinism_recurrence_rate,
                    result.ratio_laminarity_determinism,
                    labels_by_channel[electrode1][i] ## attach the label

            ])

            # Store features in rqa_matrix
            rqa_matrix[i, electrode1, electrode2, :] = features
            print(f"Stored features for segment {i+1}, ch{electrode1}_ch{electrode2}: {features}")

            rqa_results.append(result)
            print(result)







# Save the matrix
np.save('rqa_matrix_with_labels.npy', rqa_matrix)
print("rqa_matrix saved to 'rqa_matrix_with_labels.npy'")
print(f"Shape of saved matrix: {rqa_matrix.shape}")

# creation of the mean matrix.
mean_matrix = np.zeros((2, 17))

# Compute mean for each segment across electrode pairs
for i in range(2):  # Loop over segments
    # Mean of features (indices 0 to 15) across electrode pairs (axes 1 and 2)
    mean_features = np.mean(rqa_matrix[i, :, :, :16], axis=(0, 1))

    # Get the label (index 16) from the first electrode pair (assuming it's the same for all pairs)
    # Use [0, 0, 16] as a representative pair
    label = rqa_matrix[i, 0, 0, 16]

    # Store in mean_matrix
    mean_matrix[i, :16] = mean_features
    mean_matrix[i, 16] = label

# Save the mean_matrix to an .npy file
np.save('mean_rqa_matrix.npy', mean_matrix)

# Print to verify
print("Shape of mean_matrix:", mean_matrix.shape)
print("Mean matrix contents:")
print(mean_matrix)

# Optional: Verify labels
print("Labels:", mean_matrix[:, 16])
