from utils import pss, load_single_channel
import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import Pool, cpu_count
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
from collections import defaultdict
import queue

# Set start method at the very beginning
mp.set_start_method('spawn', force=True)



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



def get_opencl_for_gpu(gpu_id):
    """Create OpenCL instance for specific GPU."""
    try:
        return OpenCL(platform_id=0, device_ids=(gpu_id,))
    except Exception as e:
        print(f"Error creating OpenCL for GPU {gpu_id}: {e}. Falling back to CPU.")
        return None

def compute_batch_rqa(batch_tasks, gpu_id, opencl_instance):
    """Compute RQA for a batch of window pairs to maximize GPU utilization."""
    batch_results = []
    batch_size = len(batch_tasks)

    print(f"🔥 GPU {gpu_id}: Processing BATCH of {batch_size} computations")
    batch_start_time = time.time()

    for i, (window1, window2, electrode1, electrode2, window_idx, seg_idx, win_idx_within_seg, label) in enumerate(batch_tasks):
        if i % 10 == 0:  # Progress update every 10 computations
            print(f"GPU {gpu_id}: Batch progress: {i+1}/{batch_size}")

        min_length = min(len(window1), len(window2))
        window1 = window1[:min_length]
        window2 = window2[:min_length]

        # Skip if segments are too short
        if len(window1) < 50:
            from pyrqa.computation import RQAResult
            result = RQAResult(
                recurrence_rate=0.1, determinism=0.5, average_diagonal_line=2.0,
                longest_diagonal_line=5, divergence=0.2, entropy_diagonal_lines=1.5,
                laminarity=0.3, trapping_time=3.0, longest_vertical_line=4,
                average_white_vertical_line=2.0, longest_white_vertical_line=8,
                longest_white_vertical_line_inverse=0.125, entropy_vertical_lines=1.2,
                entropy_white_vertical_lines=1.8, ratio_determinism_recurrence_rate=5.0,
                ratio_laminarity_determinism=0.6
            )
            batch_results.append((result, electrode1, electrode2, window_idx, label))
            continue

        radius_fraction = 0.15
        norm = 'euclidean'
        max_d1, avg_d1 = pss(window1, m=3, t=1, norm=norm)
        max_d2, avg_d2 = pss(window2, m=3, t=1, norm=norm)
        mean_diameter = (max_d1 + max_d2) / 2
        radius = mean_diameter * radius_fraction

        ts1 = TimeSeries(window1, embedding_dimension=3, time_delay=1)
        ts2 = TimeSeries(window2, embedding_dimension=3, time_delay=1)

        settings = Settings(
            [ts1, ts2],
            analysis_type=Cross,
            neighbourhood=FixedRadius(radius),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1
        )

        try:
            if opencl_instance:
                computation = RQAComputation.create(settings, verbose=False, opencl=opencl_instance)
            else:
                computation = RQAComputation.create(settings, verbose=False)

            result = computation.run()
            batch_results.append((result, electrode1, electrode2, window_idx, label))

        except Exception as e:
            print(f"GPU {gpu_id}: OpenCL failed, using CPU fallback")
            computation = RQAComputation.create(settings, verbose=False)
            result = computation.run()
            batch_results.append((result, electrode1, electrode2, window_idx, label))

    batch_time = time.time() - batch_start_time
    throughput = batch_size / batch_time
    print(f"✅ GPU {gpu_id}: Batch completed - {batch_time:.2f}s, {throughput:.1f} computations/sec")

    return batch_results

def compute_rqa_for_electrode_batch(args):
    """Process multiple electrodes in large batches to maximize GPU utilization."""
    electrode_indices, windows_by_channel, labels_by_channel, window_info_by_channel, total_windows, selected_channels, channel_names, gpu_id = args

    print(f"🚀 GPU {gpu_id}: STARTING {len(electrode_indices)} electrodes: {[channel_names[i] for i in electrode_indices]}")

    # Set the GPU device
    torch.cuda.set_device(gpu_id)

    # Create OpenCL instance for THIS GPU
    opencl_instance = get_opencl_for_gpu(gpu_id)

    # Initialize partial RQA matrix for all electrodes in this batch
    partial_rqa_matrices = {
        electrode: np.zeros((total_windows, len(selected_channels), 17))
        for electrode in electrode_indices
    }

    # Create large batches of computations (50-100 computations per batch)
    BATCH_SIZE = 16000  # Increased batch size for better GPU utilization

    for electrode1 in electrode_indices:
        print(f"GPU {gpu_id}: Processing electrode {electrode1} ({channel_names[electrode1]})")

        batch_tasks = []

        for window_idx in range(total_windows):
            for electrode2 in range(len(selected_channels)):
                window1 = windows_by_channel[electrode1][window_idx]
                window2 = windows_by_channel[electrode2][window_idx]

                seg_idx, win_idx_within_seg, start_sample, end_sample = window_info_by_channel[0][window_idx]
                label = labels_by_channel[electrode1][window_idx]

                batch_tasks.append((
                    window1, window2, electrode1, electrode2,
                    window_idx, seg_idx, win_idx_within_seg, label
                ))

                # Process batch when it reaches BATCH_SIZE
                if len(batch_tasks) >= BATCH_SIZE:
                    batch_results = compute_batch_rqa(batch_tasks, gpu_id, opencl_instance)

                    # Store results
                    for result, e1, e2, w_idx, lbl in batch_results:
                        features = np.array([
                            result.recurrence_rate, result.determinism, result.average_diagonal_line,
                            result.longest_diagonal_line, result.divergence, result.entropy_diagonal_lines,
                            result.laminarity, result.trapping_time, result.longest_vertical_line,
                            result.average_white_vertical_line, result.longest_white_vertical_line,
                            result.longest_white_vertical_line_inverse, result.entropy_vertical_lines,
                            result.entropy_white_vertical_lines, result.ratio_determinism_recurrence_rate,
                            result.ratio_laminarity_determinism, lbl
                        ])
                        partial_rqa_matrices[e1][w_idx, e2, :] = features

                    batch_tasks = []  # Reset batch

        # Process any remaining tasks for this electrode
        if batch_tasks:
            batch_results = compute_batch_rqa(batch_tasks, gpu_id, opencl_instance)
            for result, e1, e2, w_idx, lbl in batch_results:
                features = np.array([
                    result.recurrence_rate, result.determinism, result.average_diagonal_line,
                    result.longest_diagonal_line, result.divergence, result.entropy_diagonal_lines,
                    result.laminarity, result.trapping_time, result.longest_vertical_line,
                    result.average_white_vertical_line, result.longest_white_vertical_line,
                    result.longest_white_vertical_line_inverse, result.entropy_vertical_lines,
                    result.entropy_white_vertical_lines, result.ratio_determinism_recurrence_rate,
                    result.ratio_laminarity_determinism, lbl
                ])
                partial_rqa_matrices[e1][w_idx, e2, :] = features

    # Clean up
    if opencl_instance:
        del opencl_instance

    print(f"✅ GPU {gpu_id}: COMPLETED {len(electrode_indices)} electrodes")
    return partial_rqa_matrices

def process_recording_optimized(metadata_path, npy_path, segment_boundaries, SEL_SIZE=512):
    """
    Optimized version that uses large batching to maximize GPU utilization.
    """
    print(f"\n=== Processing {metadata_path} / {npy_path} (OPTIMIZED) ===")
    start_time = time.time()

    # --- Load metadata ---
    metadata = np.load(metadata_path, allow_pickle=True).item()
    sfreq = metadata['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")

    # --- Define channels ---
    channel_names = [
        'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
        'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2',
        'FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8'
    ]
    selected_channels = list(range(len(channel_names)))

    # --- Load and segment channels ---
    windows_by_channel = []
    labels_by_channel = []
    window_info_by_channel = []

    for ch_idx in selected_channels:
        print(f"Loading channel {ch_idx}")
        channel_data, times = load_single_channel(npy_path, metadata_path, ch_idx)
        windows, labels, window_info = extract_non_overlapping_windows(
            channel_data, segment_boundaries, window_size=SEL_SIZE
        )
        windows_by_channel.append(windows)
        labels_by_channel.append(labels)
        window_info_by_channel.append(window_info)

    total_windows = len(windows_by_channel[0])
    print(f"\nTotal windows: {total_windows} | Channels: {len(selected_channels)}")

    # --- Initialize RQA matrix ---
    rqa_matrix = np.zeros((total_windows, len(selected_channels), len(selected_channels), 17))

    # --- Group electrodes into larger batches for each GPU ---
    num_channels = len(selected_channels)

    # Create 3 large batches (one per GPU) with multiple electrodes each
    electrodes_per_gpu = (num_channels + 2) // 3  # Round up

    gpu_batches = [
        list(range(0, electrodes_per_gpu)),
        list(range(electrodes_per_gpu, min(2 * electrodes_per_gpu, num_channels))),
        list(range(2 * electrodes_per_gpu, num_channels))
    ]

    print(f"🎯 GPU Work Distribution:")
    for i, batch in enumerate(gpu_batches):
        channel_names_str = [channel_names[idx] for idx in batch]
        print(f"  GPU {i}: {len(batch)} electrodes - {channel_names_str}")

    # Prepare arguments
    args_list = []
    for gpu_id, electrode_batch in enumerate(gpu_batches):
        if electrode_batch:  # Only add if there are electrodes to process
            args = (
                electrode_batch, windows_by_channel, labels_by_channel,
                window_info_by_channel, total_windows, selected_channels,
                channel_names, gpu_id
            )
            args_list.append(args)

    # --- Process with large batches ---
    print(f"🚀 Starting {len(args_list)} GPU workers with LARGE BATCHES...")

    completed_batches = 0
    total_batches = len(args_list)

    with Pool(processes=len(args_list)) as pool:
        for result in pool.imap_unordered(compute_rqa_for_electrode_batch, args_list):
            # result is a dict: {electrode_index: partial_rqa_matrix}
            for electrode_idx, partial_matrix in result.items():
                rqa_matrix[:, electrode_idx, :, :] = partial_matrix

            completed_batches += 1
            progress = (completed_batches / total_batches) * 100
            print(f"📊 Batch Progress: {completed_batches}/{total_batches} ({progress:.1f}%)")

    total_time = time.time() - start_time
    total_computations = total_windows * len(selected_channels) * len(selected_channels)
    throughput = total_computations / total_time

    print(f"\n✅ RQA computation completed in {total_time:.2f} seconds")
    print(f"📦 RQA matrix shape: {rqa_matrix.shape}")
    print(f"⚡ Throughput: {throughput:.1f} computations/second")

    # --- Save ---
    save_name = npy_path.replace('_filtered.npy', '.npy')
    np.save(save_name, rqa_matrix)
    print(f"💾 Saved RQA matrix to {save_name}")





def main():

    recordings = [
        {
            "metadata_path": "p5_17_metadata.npy",
            "npy_path": "p5_17_filtered.npy",
            "segment_boundaries": [
                {'start_sample': 0, 'end_sample': 2451*256, 'label': 0},
                {'start_sample': 2451*256, 'end_sample': 2571*256, 'label': 1},
                {'start_sample': 2571*256, 'end_sample': None, 'label': 0}
            ]
        },
        {
            "metadata_path": "p5_22_metadata.npy",
            "npy_path": "p5_22_filtered.npy",
            "segment_boundaries": [
                {'start_sample': 0, 'end_sample': 2348*256, 'label': 0},
                {'start_sample': 2348*256, 'end_sample': 2465*256, 'label': 1},
                {'start_sample': 2465*256, 'end_sample': None, 'label': 0}
            ]
        },
        {
            "metadata_path": "p5_22_metadata.npy",
            "npy_path": "p5_22_filtered.npy",
            "segment_boundaries": [
                {'start_sample': 0, 'end_sample': 2348*256, 'label': 0},
                {'start_sample': 2348*256, 'end_sample': 2465*256, 'label': 1},
                {'start_sample': 2465*256, 'end_sample': None, 'label': 0}
            ]
        },
    ]

    for rec in recordings:
        process_recording_optimized(
            rec["metadata_path"],
            rec["npy_path"],
            rec["segment_boundaries"],
            SEL_SIZE=512
        )

if __name__ == "__main__":
    main()


