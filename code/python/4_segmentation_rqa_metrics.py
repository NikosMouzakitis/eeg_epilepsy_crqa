import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from mpl_toolkits.mplot3d import Axes3D


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


if __name__ == "__main__":

    npy_path = "../../data/p11_06_filtered.npy"
    metadata_path = "../../data/p11_06_metadata.npy"
    SEG_SIZE = 2048
    init_idx = 90000
    # Load single channel data on an array.
    channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
    
    # Plot the single channel
    #plot_single_channel(channel_data, times, selected_channel)

    #create a 4seconds segment.
    segment_data = channel_data[init_idx:init_idx+SEG_SIZE]
    segment_time = times[init_idx:init_idx+SEG_SIZE]
    plot_single_channel(segment_data, segment_time, selected_channel)



    #embedding parameters
    m = 3
    tau = 5
    
    psv = create_phase_space_vectors(segment_data, m = m, tau = tau)

    visualize_phase_space(psv)

