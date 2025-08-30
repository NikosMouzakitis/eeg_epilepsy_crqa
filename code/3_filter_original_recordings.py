!pip install mne pyedflib
from google.colab import drive, files
import numpy as np
import pandas as pd
import mne
import pywt
import pyedflib
import matplotlib.pyplot as plt
import os
import re
import subprocess
from scipy.stats import median_abs_deviation

 edf_urls = [

                  # patient 1
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_07.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_08.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_09.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_10.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_11.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_12.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_15.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_18.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_21.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_26.edf",
                # patient 2
        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_01.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_03.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_16+.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_19.edf",
                  #patient 3
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_11.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_12.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_13.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_14.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_15.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_17.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_01.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_34.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_35.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb03/chb03_36.edf",
                #patient 4
        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_04.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_05.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_08.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb04/chb04_28.edf",
                  #patient 5
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_27.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_28.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_29.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_30.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_31.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_13.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_17.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb05/chb05_22.edf",
                  #patient 6
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_05.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_07.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_08.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_15.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_17.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_01.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_09.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_10.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_13.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_18.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb06/chb06_24.edf",
                  #patient 7
        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_04.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_12.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_13.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb07/chb07_19.edf",
                  # patient 8
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_15.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_16.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_17.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_18.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_19.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_05.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_11.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_13.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb08/chb08_21.edf",

        #
                  #patient 9
        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_08.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_19.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb09/chb09_15.edf",

                 #patient 10
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_12.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_20.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_27.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_30.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_31.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_38.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_89.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_01.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_02.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_03.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_05.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb10/chb10_07.edf",

                  #patient 11
        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_82.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_92.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_99.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_04.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_05.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb11/chb11_06.edf",


                  #patient 12

        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_06.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_08.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_09.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_10.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_11.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_23.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_27.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_28.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_29.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_33.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_36.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_38.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_42.edf",

        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_19.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_20.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_21.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_24.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_32.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_34.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_35.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_37.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_40.edf",
        "https://physionet.org/files/chbmit/1.0.0/chb12/chb12_41.edf",
        #----missing 3 to be perfectly balanced.


                  #patient 13


                  #patient 14


                  #patient 15


                  #patient 16


                  #patient 17


                  #patient 18


                  #patient 19


                  #patient 20


                  #patient 21


                  #patient 22



    ]

# Mount Google Drive
drive.mount('/content/drive')

# Define desired channels (final list for CSV, excluding T8-P8-1)
desired_channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
]

def apply_wavelet_filter_minmax(signal, wavelet='sym8', level=4, threshold_scale=0.5, mode='hard', sfreq=None, apply_normalization=True):
    """Apply wavelet denoising, second bandpass filter, and optionally Min-Max normalize to [-1, 1]."""
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    sigma = median_abs_deviation(coeffs[-1]) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[2:]]
    denoised = pywt.waverec(coeffs, wavelet)
    denoised = denoised[:len(signal)]

    raw_temp = mne.io.RawArray(denoised[np.newaxis, :], mne.create_info(['temp'], sfreq, ch_types='eeg'))
    raw_temp.filter(l_freq=1.0, h_freq=60.0, method='iir', iir_params=dict(order=4, ftype='butter'), phase='zero', verbose=False)
    denoised = raw_temp.get_data()[0]

    if apply_normalization:
        min_val, max_val = denoised.min(), denoised.max()
        if max_val > min_val:
            normalized = 2 * (denoised - min_val) / (max_val - min_val) - 1
        else:
            normalized = np.zeros_like(denoised)
        return normalized
    return denoised

def is_valid_edf(file_path):
    """Check if a file is a valid EDF file by inspecting its header."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            return header.startswith(b'0')
    except Exception:
        return False

def load_eeg_data(edf_path):
    """Load EEG data and apply initial bandpass filter (0.5–60 Hz)."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = float(raw.info['sfreq'])
        raw.filter(l_freq=0.5, h_freq=60.0, method='iir', iir_params=dict(order=4, ftype='butter'), phase='zero', verbose=False)
        print(f"Applied bandpass filter (0.5–60 Hz) to {edf_path}")
        return raw, sfreq
    except Exception as e:
        print(f"Failed to load {edf_path}: {e}")
        return None, None

def extract_patient_number(edf_filename):
    """Extract patient number and recording number from EDF filename."""
    match = re.match(r'chb(\d+)([_+][\w+]+)?\.edf', edf_filename)
    if match:
        patient_num = int(match.group(1))
        recording_num = match.group(2)[1:] if match.group(2) else ""
        return f'p{patient_num}_{recording_num}' if recording_num else f'p{patient_num}'
    print(f"Failed to parse filename: {edf_filename}")
    return None

def download_edf_file(url, output_dir, username=None, password=None):
    """Download EDF file from URL to output_dir using wget with cookies and authentication."""
    filename = os.path.basename(url)
    local_path = os.path.join(output_dir, filename)
    cookie_file = os.path.join(output_dir, 'cookies.txt')

    # Remove existing file if invalid to force re-download
    if os.path.exists(local_path) and not is_valid_edf(local_path):
        print(f"Removing invalid existing file: {local_path}")
        os.remove(local_path)

    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path} using wget")
        try:
            # Initialize session to get cookies
            cmd_init = ['wget', '--quiet', '--save-cookies', cookie_file, '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'https://physionet.org/']
            if username and password:
                cmd_init.extend(['--user', username, '--password', password])
            result_init = subprocess.run(cmd_init, capture_output=True, text=True)
            if result_init.returncode != 0:
                print(f"Failed to initialize session for {url}: {result_init.stderr}")
                return None

            # Download the file using cookies
            cmd = ['wget', '--quiet', '--load-cookies', cookie_file, '--save-cookies', cookie_file, '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', '--no-check-certificate', url, '-O', local_path]
            if username and password:
                cmd.extend(['--user', username, '--password', password])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"wget failed for {url}: {result.stderr}")
                if os.path.exists(cookie_file):
                    os.remove(cookie_file)
                return None

            # Debug: Print first 100 bytes
            with open(local_path, 'rb') as f:
                first_bytes = f.read(100)
                print(f"First 100 bytes of {local_path}: {first_bytes}")
            if not is_valid_edf(local_path):
                print(f"Downloaded file {local_path} is not a valid EDF file")
                os.remove(local_path)
                if os.path.exists(cookie_file):
                    os.remove(cookie_file)
                return None
            print(f"Downloaded valid EDF file: {local_path}")
            if os.path.exists(cookie_file):
                os.remove(cookie_file)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            if os.path.exists(cookie_file):
                os.remove(cookie_file)
            return None
    else:
        if not is_valid_edf(local_path):
            print(f"Existing file {local_path} is not a valid EDF file")
            return None
        print(f"File {local_path} already exists and is valid, skipping download")
    return local_path

def manual_upload_fallback(output_dir):
    """Prompt user to manually upload EDF files in Colab."""
    print("Automated download failed. Please upload EDF files manually.")
    uploaded = files.upload()
    for filename, content in uploaded.items():
        local_path = os.path.join(output_dir, filename)
        with open(local_path, 'wb') as f:
            f.write(content)
        print(f"Uploaded {filename} to {local_path}")
        if not is_valid_edf(local_path):
            print(f"Uploaded file {local_path} is not a valid EDF file")
            os.remove(local_path)
            return None
        return local_path
    return None

def process_and_save_npy(edf_path, output_dir, file_id):
    """Process EDF file, filter desired channels, rename T8-P8-0 or T8-P8 to T8-P8, drop T8-P8-1 and non-EEG channels, and save to .npy."""
    raw, sfreq = load_eeg_data(edf_path)
    if raw is None:
        return None, None, None
    channels = raw.ch_names

    print(f"Original channels in {edf_path}: {channels}")
    available_channels = []
    for ch in channels:
        if ch in ['T8-P8-0', 'T8-P8']:
            available_channels.append('T8-P8')
        elif ch == 'T8-P8-1' or ch.startswith('-'):
            print(f"Dropping channel: {ch}")
            continue
        elif ch in desired_channels:
            available_channels.append(ch)
        else:
            print(f"Dropping non-EEG or undesired channel: {ch}")

    channels_to_keep = [ch for ch in desired_channels if ch in available_channels]
    if not channels_to_keep:
        print(f"No desired channels found in {edf_path}, skipping")
        return None, None, None

    pick_channels = []
    for ch in channels_to_keep:
        if ch == 'T8-P8':
            if 'T8-P8-0' in channels and 'T8-P8-0' not in pick_channels:
                pick_channels.append('T8-P8-0')
            elif 'T8-P8' in channels and 'T8-P8' not in pick_channels:
                pick_channels.append('T8-P8')
        elif ch not in pick_channels:
            pick_channels.append(ch)

    try:
        raw.pick(pick_channels)
    except ValueError as e:
        print(f"Error picking channels in {edf_path}: {e}")
        return None, None, None

    data = raw.get_data()
    channels_to_keep = [ch if ch != 'T8-P8-0' else 'T8-P8' for ch in raw.ch_names]
    print(f"Selected channels for processing: {channels_to_keep}")

    n_channels, n_samples = data.shape
    filtered = np.empty_like(data, dtype=float)
    for i in range(n_channels):
        filtered[i] = apply_wavelet_filter_minmax(data[i], sfreq=sfreq)

    # Ensure output array matches desired_channels order
    final_data = np.zeros((len(desired_channels), n_samples), dtype=float)
    final_channels = desired_channels.copy()
    for i, ch in enumerate(channels_to_keep):
        if ch in desired_channels:
            idx = desired_channels.index(ch)
            final_data[idx] = filtered[i]

    # Save data as .npy
    output_npy = os.path.join(output_dir, f"{file_id}_filtered.npy")
    np.save(output_npy, final_data)
    print(f"Saved filtered matrix to {output_npy}, shape={final_data.shape}, channels={final_channels}")

    # Save metadata (channels, sfreq, times) as a separate .npy file
    times = np.arange(n_samples) / sfreq
    metadata = {
        'channels': final_channels,
        'sfreq': sfreq,
        'times': times
    }
    output_metadata = os.path.join(output_dir, f"{file_id}_metadata.npy")
    np.save(output_metadata, metadata)
    print(f"Saved metadata to {output_metadata}")

    return output_npy, final_channels, sfreq


def plot_first5_from_csv(csv_path, save_fig):
    """Plot first 5 channels from CSV and save plot."""
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
    fig.suptitle(f"First {n_plot} Channels (SYM8 L4, hard thr=0.5, BP 0.5–60 Hz, 1–60 Hz)", y=0.95)
    plt.tight_layout()
    plt.savefig(save_fig, dpi=150)
    print(f"Plot saved to {save_fig}")
    plt.show()

def process_multiple_edf_files(edf_urls, drive_output_dir, username=None, password=None):
    """Process multiple EDF files from URLs or local paths and save results to Google Drive."""
    os.makedirs(drive_output_dir, exist_ok=True)

    for url in edf_urls:
        edf_filename = os.path.basename(url)
        # Handle local files
        if url.startswith('/'):
            edf_path = url
            if not is_valid_edf(edf_path):
                print(f"Local file {edf_path} is not a valid EDF file")
                continue
        else:
            edf_path = download_edf_file(url, drive_output_dir, username, password)
            if not edf_path:
                print(f"Download failed for {edf_filename}. Attempting manual upload...")
                edf_path = manual_upload_fallback(drive_output_dir)
                if not edf_path:
                    print(f"Skipping {edf_filename} due to download or validation failure")
                    continue

        file_id = extract_patient_number(edf_filename)
        if not file_id:
            print(f"Skipping {edf_filename} due to file ID extraction failure")
            continue

        try:
            csv_path, channels, sfreq = process_and_save_npy(
                edf_path, drive_output_dir, file_id
            )

            if csv_path is None:
                print(f"Failed to process {edf_filename}, skipping")
                continue

            #plot_path = os.path.join(drive_output_dir, f"{file_id}_filtered_plot.png")
            #plot_first5_from_csv(csv_path, plot_path)
        except Exception as e:
            print(f"Error processing {edf_filename}: {e}")
            continue

        # Delete the EDF file after processing (skip for local files)
        if not url.startswith('/'):
            try:
                if os.path.exists(edf_path):
                    os.remove(edf_path)
                    print(f"Deleted EDF file: {edf_path}")
                else:
                    print(f"EDF file not found, cannot delete: {edf_path}")
            except Exception as e:
                print(f"Error deleting EDF file {edf_path}: {e}")

if __name__ == "__main__":
    output_dir = "/content/drive/My Drive/EEG_Processed"
    physionet_username = ""
    physionet_password = ""

    process_multiple_edf_files(edf_urls, output_dir, physionet_username, physionet_password)
