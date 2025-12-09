import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Settings
# ----------------------------
data_folder = os.getcwd()  # current directory
plots_folder = os.path.join(data_folder, "per_patient_determinism")
os.makedirs(plots_folder, exist_ok=True)

# ----------------------------
# Load .npy files
# ----------------------------
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
print(f"Found {len(npy_files)} .npy files:")
for f in npy_files:
    print(" -", f)

# ----------------------------
# Process each file (patient)
# ----------------------------
for file_name in npy_files:
    file_path = os.path.join(data_folder, file_name)
    
    # Extract patient ID
    match = re.match(r'(p\d+)_', file_name)
    patient_id = match.group(1) if match else "unknown"
    
    print(f"\nProcessing {file_name} (Patient {patient_id}) ...")
    
    # Load data
    data = np.load(file_path)  # shape: (N_samples, 22, 22, 17)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract labels
    labels = data[:, 0, 0, -1].round().astype(int)  # last dim = label
    
    # Extract Determinism feature (index 1)
    det_data = data[:, :, :, 1]  # shape: (N_samples, 22, 22)
    
    # Split normal and epileptic
    det_normal = det_data[labels == 0]
    det_epileptic = det_data[labels == 1]
    
    # Compute mean across samples
    if det_normal.shape[0] > 0:
        mean_normal = det_normal.mean(axis=0)
    else:
        mean_normal = np.zeros((22,22))
        print("Warning: No normal samples for this patient")
        
    if det_epileptic.shape[0] > 0:
        mean_epileptic = det_epileptic.mean(axis=0)
    else:
        mean_epileptic = np.zeros((22,22))
        print("Warning: No epileptic samples for this patient")
    
    # ----------------------------
    # Plot per-patient
    # ----------------------------
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.imshow(mean_normal, cmap='gray', origin='lower')
    plt.colorbar(label='Determinism (DET)')
    plt.title(f'Patient {patient_id} - Normal (Label 0)')
    plt.xlabel('Electrode X')
    plt.ylabel('Electrode Y')
    
    plt.subplot(1,2,2)
    plt.imshow(mean_epileptic, cmap='gray', origin='lower')
    plt.colorbar(label='Determinism (DET)')
    plt.title(f'Patient {patient_id} - Epileptic (Label 1)')
    plt.xlabel('Electrode X')
    plt.ylabel('Electrode Y')
    
    plt.tight_layout()
    save_path = os.path.join(plots_folder, f"{patient_id}_determinism_mean.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved Determinism heatmaps for Patient {patient_id} -> {save_path}")

print(f"\nAll per-patient plots saved in: {plots_folder}")

