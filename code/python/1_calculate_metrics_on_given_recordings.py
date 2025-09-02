import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
from scipy.stats import median_abs_deviation
from scipy.signal import welch
import pandas as pd

NUM_CHAN = 2
PLOT_START = 40
PLOT_END = 44
'''

File used in order to get the metrics 
        'SNR (dB)'
        'RMSE (μV)'
        'NRMSE (%)'
        'Correlation'
        'PRD (%)'
        for a number of given edf recordings


'''
def load_eeg_data(edf_path, num_channels=NUM_CHAN, l_freq=0.5, h_freq=60.0):
    try:
        temp_raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        unique_ch_names = list(dict.fromkeys(temp_raw.ch_names))
        channels = unique_ch_names
        #channels = unique_ch_names[:num_channels]
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick(channels)
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
                 iir_params=dict(order=4, ftype='butter'),
                 phase='zero', verbose=True)
        print(f"Applied bandpass filter ({l_freq}–{h_freq} Hz)")
        return raw, channels, raw.info['sfreq']
    except Exception as e:
        print(f"Error loading EDF file: {str(e)}")
        return None, None, None

def wavelet_denoise(signal, wavelet='sym4', mode='hard', level=7, threshold_scale=2.0):
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = median_abs_deviation(coeffs[-1]) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1] = np.zeros_like(coeffs[1])
        coeffs[2:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[2:]]
        return pywt.waverec(coeffs, wavelet)
    except Exception as e:
        print(f"Error in wavelet denoising: {e}")
        return signal

def bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=60.0):
    try:
        raw_temp = mne.io.RawArray(signal[np.newaxis, :],
                                  mne.create_info(['temp'], sfreq, ch_types='eeg'))
        raw_temp.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
                       iir_params=dict(order=4, ftype='butter'),
                       phase='zero', verbose=True)
        return raw_temp.get_data()[0]
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return signal

def calculate_residual_snr(raw_signal, denoised_signal):
    try:
        residual = raw_signal - denoised_signal
        signal_power = np.mean(denoised_signal ** 2)
        noise_power = np.mean(residual ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-12))
    except Exception as e:
        print(f"Error in SNR calculation: {e}")
        return np.nan

def calculate_rmse(original, processed):
    return np.sqrt(np.mean((original - processed)**2))

def calculate_nrmse(original, processed):
    rmse = calculate_rmse(original, processed)
    signal_range = np.max(original) - np.min(original)
    return (rmse / signal_range) * 100

def calculate_correlation(original, processed):
    return np.corrcoef(original, processed)[0, 1]

def calculate_prd(original, processed):
    numerator = np.sum((original - processed)**2)
    denominator = np.sum(original**2)
    return 100 * np.sqrt(numerator / denominator)

def calculate_all_metrics(original, processed):
    return {
        'SNR (dB)': calculate_residual_snr(original, processed),
        'RMSE (μV)': calculate_rmse(original, processed),
        'NRMSE (%)': calculate_nrmse(original, processed),
        'Correlation': calculate_correlation(original, processed),
        'PRD (%)': calculate_prd(original, processed)
    }

def process_files(edf_paths):
    all_rows = []
    example_data = None  # first file, first channel original/filtered for plotting
    sfreq_example = None
    times_example = None

    for file_idx, path in enumerate(edf_paths):
        raw, channels, sfreq = load_eeg_data(path)
        if raw is None:
            continue

        full_data = raw.get_data(picks=channels)
        start_sample = int(PLOT_START * sfreq)
        stop_sample = int(PLOT_END * sfreq)

        if file_idx == 0:
            sfreq_example = sfreq
            times_example = np.arange(start_sample, stop_sample) / sfreq + PLOT_START
            example_data = full_data[0, start_sample:stop_sample]  # first channel

        wavelets = ['db4', 'db6', 'sym4', 'sym8', 'coif3', 'bior3.5']
        levels = [2, 3, 4, 5]
        threshold_scales = [0.5, 1.0, 2.0, 3.0, 4.0]
        modes = ['hard', 'soft']

        filters = []
        for wavelet in wavelets:
            for level in levels:
                for thresh in threshold_scales:
                    for mode in modes:
                        name = f"{wavelet.upper()} LVL:{level} THR:{thresh} {mode}"
                        filters.append(
                            (name, lambda x, w=wavelet, l=level, t=thresh, m=mode:
                                bandpass_filter(
                                    wavelet_denoise(x, wavelet=w, level=l, threshold_scale=t, mode=m),
                                    sfreq, l_freq=1.0, h_freq=60.0
                                )
                            )
                        )

        for filt_name, filt_func in filters:
            for ch_idx, ch_name in enumerate(channels):
                original = full_data[ch_idx]
                processed = filt_func(original)
                metrics = calculate_all_metrics(original, processed)

                all_rows.append({
                    "File": path,
                    "Filter": filt_name,
                    "Channel": ch_name,
                    **metrics
                })

                if file_idx == 0 and ch_idx == 0:
                    # Store processed example data for first file/channel
                    yield_data = metrics.copy()
                    yield_data["processed"] = processed[start_sample:stop_sample]
                    yield_data["filter_name"] = filt_name
                    yield (yield_data, example_data, sfreq_example, times_example)

    df_metrics = pd.DataFrame(all_rows)
    df_metrics.to_csv("eeg_metrics_results.csv", index=False)
    print("Metrics saved to eeg_metrics_results.csv")
    return df_metrics

def plot_top_filters(df_metrics, example_data, processed_examples, sfreq_example, times_example):
    metrics_list = ['SNR (dB)', 'RMSE (μV)', 'NRMSE (%)', 'Correlation', 'PRD (%)']
    better_high = {'SNR (dB)': True, 'Correlation': True,
                   'RMSE (μV)': False, 'NRMSE (%)': False, 'PRD (%)': False}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_list):
        avg_scores = df_metrics.groupby("Filter")[metric].mean()
        avg_scores = avg_scores.sort_values(ascending=not better_high[metric])
        top_filters = avg_scores.head(6).index


        print(f"\n=== Top 6 by {metric} ===")
        for rank, (fname, score) in enumerate(avg_scores.head(6).items(), start=1):
            print(f"{rank}. {fname:30s} {score:.4f}")

        ax = axes[idx]
        ax.plot(times_example, example_data / np.std(example_data), label="Original", lw=2)

        for filt_name in top_filters:
            proc = processed_examples[filt_name]
            ax.plot(times_example, proc / np.std(proc), label=filt_name, alpha=0.7)

        ax.set_title(f"Top 6 by {metric}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (norm.)")
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    EDF_PATHS = [
        "chb01_03.edf", "chb02_16.edf", "chb03_34.edf", "chb04_05.edf", "chb08_13.edf"
      ]

    processed_examples = {}
    example_data = None
    sfreq_example = None
    times_example = None

    # First pass to process files and collect example processed data
    gen = process_files(EDF_PATHS)
    for result, orig_data, sfreq_ex, times_ex in gen:
        if example_data is None:
            example_data = orig_data
            sfreq_example = sfreq_ex
            times_example = times_ex
        processed_examples[result["filter_name"]] = result["processed"]

    # Load metrics file
    df_metrics = pd.read_csv("eeg_metrics_results.csv")

    # Plot
    plot_top_filters(df_metrics, example_data, processed_examples, sfreq_example, times_example)


