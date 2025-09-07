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

if __name__ == "__main__":
    npy_path = "../../data/p11_06_filtered.npy"
    metadata_path = "../../data/p11_06_metadata.npy"
    
    # Load single channel data on an array.
    channel_data, times = load_single_channel(npy_path, metadata_path, selected_channel)
    
    # Plot the single channel
    #plot_single_channel(channel_data, times, selected_channel)

    #create a 4seconds segment.
    segment_data = channel_data[0:1024]
    segment_time = times[0:1024]
    plot_single_channel(segment_data, segment_time, selected_channel)

