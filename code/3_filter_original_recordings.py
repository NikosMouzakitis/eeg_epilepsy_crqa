import numpy as np
import pandas as pd
import mne
import pywt
import matplotlib.pyplot as plt

def apply_wavelet_filter_minmax(signal, wavelet='sym8', level=4, threshold_scale=0.5, mode='hard'):
    """Apply wavelet denoising and then Min-Max normalize to [-1, 1]."""
    # --- Wavelet denoising ---
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    universal_thr = sigma * np.sqrt(2 * np.log(len(signal)))
    thr = threshold_scale * universal_thr
    coeffs[1:] = [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    denoised = denoised[:len(signal)]  # match length

    # --- Min-Max normalization to [-1, 1] ---
    min_val, max_val = denoised.min(), denoised.max()
    if max_val > min_val:  
        normalized = 2 * (denoised - min_val) / (max_val - min_val) - 1
    else:
        normalized = np.zeros_like(denoised)  # flat signal

    return normalized

def load_eeg_data(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    channels = list(dict.fromkeys(raw.ch_names))
    sfreq = float(raw.info['sfreq'])
    data = raw.get_data(picks=channels)
    return data, channels, sfreq

def process_and_save_csv(edf_path, output_csv):
    data, channels, sfreq = load_eeg_data(edf_path)
    n_channels, n_samples = data.shape
    filtered = np.empty_like(data, dtype=float)

    for i in range(n_channels):
        filtered[i] = apply_wavelet_filter_minmax(data[i])

    filtered_T = filtered.T
    times = np.arange(n_samples) / sfreq
    df = pd.DataFrame(filtered_T, columns=channels)
    df.insert(0, "Time(s)", times)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved filtered matrix to {output_csv}, shape={filtered_T.shape}")
    return output_csv, channels, sfreq

def plot_first5_from_csv(csv_path, save_fig="filtered_plot.png"):
    df = pd.read_csv(csv_path)
    times = df["Time(s)"].values
    all_channels = [c for c in df.columns if c != "Time(s)"]
    n_plot = min(5, len(all_channels))

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 10), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        ch = all_channels[i]
        axes[i].plot(times, df[ch].values, linewidth=0.8)
        axes[i].set_ylabel(ch)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"First {n_plot} Channels (SYM8 L4, hard thr=0.5)", y=0.95)

    plt.tight_layout()
    plt.savefig(save_fig, dpi=150)
    print(f"✅ Plot saved to {save_fig}")
    plt.show()

if __name__ == "__main__":
    edf_file = "chb01_03.edf"   # change this path
    csv_path, channels, sfreq = process_and_save_csv(edf_file, "sym8_lvl4_thr0.5_hard.csv")
    plot_first5_from_csv(csv_path)

