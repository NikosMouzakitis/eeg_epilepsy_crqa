import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import cpu_count
import subprocess
import time
from datetime import datetime
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.opencl import OpenCL

from pyrqa.computation import RQAComputation, RPComputation
import pyopencl as cl

# Previous functions implementation in utils file. keeping only RQA/CRQA here.
from utils import find_optimal_tau_ami, false_nearest_neighbors, pss, save_rp_plot, load_single_channel

opencl = OpenCL(platform_id=0, device_ids=(0,1))
 
def get_nvidia_smi_output():
    """Run nvidia-smi and return its output."""
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        return output
    except subprocess.CalledProcessError as e:
        return f"Error running nvidia-smi: {e}"

def get_opencl_context(gpu_id):
    """Create an OpenCL context for the specified GPU."""
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("No OpenCL platforms found. Falling back to CPU.")
            return None
        # Select NVIDIA platform (assuming NVIDIA GPUs)
        nvidia_platform = None
        for platform in platforms:
            if "NVIDIA" in platform.name:
                nvidia_platform = platform
                break
        if not nvidia_platform:
            print("No NVIDIA OpenCL platform found. Falling back to CPU.")
            return None
        devices = nvidia_platform.get_devices(device_type=cl.device_type.GPU)
        if gpu_id >= len(devices):
            print(f"GPU {gpu_id} not found. Only {len(devices)} GPUs available. Falling back to CPU.")
            return None
        device = devices[gpu_id]
        print(f"Using OpenCL device: {device.name} (GPU {gpu_id})")
        print(device)
        context = cl.Context(devices=[device])
        print("context" )
        print(context)
        return context
    except Exception as e:
        print(f"Error setting up OpenCL context for GPU {gpu_id}: {e}. Falling back to CPU.")
        return None


def compute_rqa_for_electrode1(args, gpu_id):
    """Worker function for each electrode, pinned to a specific GPU."""
    electrode1, windows_by_channel, labels_by_channel, window_info_by_channel, total_windows, selected_channels, channel_names = args
    print(f"Starting process for electrode1={electrode1} on GPU {gpu_id}")
    
    # Set the GPU device for any PyTorch operations
    torch.cuda.set_device(gpu_id)
    
    # Create OpenCL context for this GPU
    opencl_context = get_opencl_context(gpu_id)
    
    # Initialize partial RQA matrix
    partial_rqa_matrix = np.zeros((total_windows, len(selected_channels), 17))
    
    for window_idx in range(total_windows):
        for electrode2 in range(len(selected_channels)):
            window1 = windows_by_channel[electrode1][window_idx]
            window2 = windows_by_channel[electrode2][window_idx]
            
            seg_idx, win_idx_within_seg, start_sample, end_sample = window_info_by_channel[0][window_idx]
            print(f"Computing Cross RQA for window {window_idx+1}/{total_windows} "
                  f"(Segment {seg_idx}, Window {win_idx_within_seg}) "
                  f"(Channel {electrode1} vs Channel {electrode2}) on GPU {gpu_id}")
            
            # Find optimal parameters
            max_tau_to_test = 25
            s1_otau, _ = find_optimal_tau_ami(window1, max_tau=max_tau_to_test)
            s2_otau, _ = find_optimal_tau_ami(window2, max_tau=max_tau_to_test)
            optimal_tau = min(s1_otau, s2_otau)
            fnn_ratio, s1_odim = false_nearest_neighbors(window1, tau=optimal_tau, max_dim=10, rtol=15.0, atol=2.0)
            fnn_ratio, s2_odim = false_nearest_neighbors(window2, tau=optimal_tau, max_dim=10, rtol=15.0, atol=2.0)
            optimal_dim = min(s1_odim, s2_odim)
            print(f"optimal_tau: {optimal_tau}, optimal_dim: {optimal_dim}")
            
            ch1_name = channel_names[electrode1] if channel_names else f"Ch{electrode1}"
            ch2_name = channel_names[electrode2] if channel_names else f"Ch{electrode2}"
            
            result = compute_cross_rqa(
                window1, window2,
                embedding_dimension=optimal_dim,
                time_delay=optimal_tau,
                channel1_name=ch1_name,
                channel2_name=ch2_name,
                segment_idx=f"seg{seg_idx}_win{win_idx_within_seg}",
                save_rp=False,
           #     opencl_context=opencl_context
            )
            
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
                labels_by_channel[electrode1][window_idx]
            ])
            
            print(f"Features for window {window_idx+1}, ch{electrode1}_ch{electrode2}: {features[:5]}...")
            partial_rqa_matrix[window_idx, electrode2, :] = features
    
    # Release OpenCL context
    if opencl_context:
        del opencl_context
    
    return electrode1, partial_rqa_matrix

def extract_non_overlapping_windows(channel_data, segment_boundaries, window_size=256):
    all_windows = []
    all_labels = []
    window_info = []
    
    for seg_idx, boundary in enumerate(segment_boundaries):
        start_sample = boundary['start_sample']
        end_sample = boundary['end_sample'] if boundary['end_sample'] is not None else len(channel_data)
        label = boundary['label']
        
        if start_sample < len(channel_data) and end_sample <= len(channel_data) and start_sample < end_sample:
            segment_length = end_sample - start_sample
            num_windows = segment_length // window_size
            
            for window_idx in range(num_windows):
                window_start = start_sample + window_idx * window_size
                window_end = window_start + window_size
                window_data = channel_data[window_start:window_end]
                all_windows.append(window_data)
                all_labels.append(label)
                window_info.append((seg_idx, window_idx, start_sample, end_sample))
                
            print(f"Segment {seg_idx} ({'Epileptic' if label == 1 else 'Normal'}): "
                  f"extracted {num_windows} windows from {segment_length} samples "
                  f"(remaining: {segment_length % window_size} samples)")
        else:
            print(f"Invalid segment {seg_idx}: samples {start_sample} to {end_sample}")
    
    return all_windows, all_labels, window_info

def monitor_gpus(duration, interval=10):
    """Monitor GPU utilization during the computation."""
    start_time = time.time()
    while time.time() - start_time < duration:
        print("\n" + "="*80)
        print(f"GPU Status at {datetime.now()}:")
        print(get_nvidia_smi_output())
        print("="*80 + "\n")
        time.sleep(interval)


def compute_cross_rqa(segment1, segment2, embedding_dimension=3, time_delay=1, channel1_name="Channel1", channel2_name="Channel2", segment_idx=0, save_rp=False, gpu_id=0):
    print(f"Computing CRQA in process {mp.current_process().name}")
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
    
    print(f"Running RQA metrics computation m={embedding_dimension} tau={time_delay}")
    
    try:
        computation = RQAComputation.create(settings, verbose=False, opencl=opencl)
        result = computation.run()
        print("OpenCL computation successful")
    except Exception as e:
       print(f"OpenCL computation failed: {e}. Falling back to CPU.")
       while True:
           x =9

    if save_rp:
        print("Calculating and saving RP plot")
        try:
            rp_computation = RPComputation.create(settings, cuda=True)
            rp_result = rp_computation.run()
            save_rp_plot(rp_result, channel1_name, channel2_name, f"seg{segment_idx}")
        except Exception as e:
            print(f"CUDA RP computation failed: {e}. Falling back to CPU.")
            rp_computation = RPComputation.create(settings)
            rp_result = rp_computation.run()
            save_rp_plot(rp_result, channel1_name, channel2_name, f"seg{segment_idx}")

    return result

def main():
    # Configuration
    metadata_path = 'p24_04_metadata.npy'  # Update this
    npy_path = 'p24_04_filtered.npy'          # Update this
    SEL_SIZE = 512
    duration = 10*3600  # Maximum runtime in seconds (1 hour, adjust as needed)
    
    # Segment boundaries
    segment_boundaries = [
#        {'start_sample': 0, 'end_sample': 278528, 'label': 0},      # Normal: 0–1088 seconds
        {'start_sample': 278528, 'end_sample': 286720, 'label': 1}, # Seizure 1: 1088–1120 seconds
#        {'start_sample': 286720, 'end_sample': 361216, 'label': 0}, # Normal: 1120–1411 seconds
        {'start_sample': 361216, 'end_sample': 368128, 'label': 1}, # Seizure 2: 1411–1438 seconds
#        {'start_sample': 368128, 'end_sample': 446720, 'label': 0}, # Normal: 1438–1745 seconds
        {'start_sample': 446720, 'end_sample': 451584, 'label': 1}, # Seizure 3: 1745–1764 seconds
#        {'start_sample': 451584, 'end_sample': None, 'label': 0}    # Normal: 1764 seconds–end
    ] 
    # Channel names
    channel_names = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
    ]
    selected_channels = list(range(len(channel_names)))
    
    # Load metadata
    metadata = np.load(metadata_path, allow_pickle=True).item()
    sfreq = metadata['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")
    
    # Process channels
    windows_by_channel = []
    labels_by_channel = []
    window_info_by_channel = []
    
    for ch_idx in selected_channels:
        print(f"\nProcessing channel {ch_idx}")
        channel_data, times = load_single_channel(npy_path, metadata_path, ch_idx)
        print(f"Channel {ch_idx} - Loaded channel data with shape: {channel_data.shape}")
        
        windows, labels, window_info = extract_non_overlapping_windows(
            channel_data, segment_boundaries, window_size=SEL_SIZE
        )
        
        windows_by_channel.append(windows)
        labels_by_channel.append(labels)
        window_info_by_channel.append(window_info)
        print(f"Channel {ch_idx} - Total windows extracted: {len(windows)}")
    
    total_windows = len(windows_by_channel[0])
    print(f"\nTotal windows to process: {total_windows}")
    print(f"Total channels: {len(selected_channels)}")
    
    # Initialize RQA matrix
    rqa_matrix = np.zeros((total_windows, len(selected_channels), len(selected_channels), 17))
    
    # Prepare arguments for each process
    args_list = [
        ((electrode1, windows_by_channel, labels_by_channel, window_info_by_channel, total_windows, selected_channels, channel_names), gpu_id)
        for electrode1, gpu_id in zip(range(len(selected_channels)), [0, 1] * (len(selected_channels) // 2 + 1))
    ]
    
    # Start GPU monitoring in a separate process
    mp.set_start_method('spawn', force=True)  # Required for CUDA
#    monitor_process = mp.Process(target=monitor_gpus, args=(duration,))
#    monitor_process.start()
    
    # Create a process pool with GPU-aware processes
    num_processes = min(len(selected_channels), cpu_count())
    print(f"Using {num_processes} processes across 2 GPUs")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_rqa_for_electrode1, args_list)
        
        # Collect results
        for electrode1, partial_rqa_matrix in results:
            rqa_matrix[:, electrode1, :, :] = partial_rqa_matrix
    
    
    print("RQA computation completed")
    print(f"Final shape of rqa_matrix: {rqa_matrix.shape}")
    
    # Save the matrix
    np.save('p24_04_epileptic512.npy', rqa_matrix)
    print("rqa_matrix saved to 'rqa_matrix_all_windows.npy'")
    
    # Print final GPU status
    print("\nFinal GPU Status:")
    print(get_nvidia_smi_output())

if __name__ == "__main__":
    main()




