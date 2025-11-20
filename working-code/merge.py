import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def merge_normal_segments(data, window_size=5):
    """
    Merges consecutive NORMAL segments (label=0) into windows of N segments.
    Epileptic segments remain unchanged.

    Args:
        data: np.ndarray shaped (N, X, Y, 17)
        window_size: default 5

    Returns:
        merged_data: np.ndarray with merged normal segments
    """

    # Clean invalid numbers
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    labels = data[:, 0, 0, 16].round().astype(int)

    merged_segments = []
    i = 0
    N = len(data)

    while i < N:
        # If epileptic → keep as-is
        if labels[i] == 1:
            merged_segments.append(data[i])
            i += 1
            continue

        # Else: collect a continuous normal run
        start = i
        while i < N and labels[i] == 0:
            i += 1
        end = i  # non-inclusive

        normal_block = data[start:end]  # shape = (K, X, Y, 17)
        block_len = len(normal_block)

        # Now merge in windows of "window_size"
        full_windows = block_len // window_size
        remainder = block_len % window_size

        # Merge full windows
        for w in range(full_windows):
            chunk = normal_block[w*window_size : (w+1)*window_size]
            merged = chunk.mean(axis=0)
            merged[:, :, 16] = 0   # enforce label
            merged_segments.append(merged)

        # Handle remainder
        if remainder > 0:
            chunk = normal_block[-remainder:]
            merged = chunk.mean(axis=0)
            merged[:, :, 16] = 0
            merged_segments.append(merged)

    return np.array(merged_segments)


# ----------------------------
# Settings
# ----------------------------
data_folder = os.getcwd()   # all .npy files are in current working directory
plots_folder = os.path.join(data_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# Feature names (16 RQA features)
rqa_feature_names = [
    "Recurrence rate (RR)",
    "Determinism (DET)",
    "Average diagonal line length (L)",
    "Longest diagonal line length (L_max)",
    "Divergence (DIV)",
    "Entropy diagonal lines (L_entr)",
    "Laminarity (LAM)",
    "Trapping time (TT)",
    "Longest vertical line length (V_max)",
    "Entropy vertical lines (V_entr)",
    "Average white vertical line length (W)",
    "Longest white vertical line length (W_max)",
    "Longest white vertical line length divergence (W_div)",
    "Entropy white vertical lines (W_entr)",
    "DET/RR",
    "LAM/DET"
]

# ----------------------------
# Load .npy files
# ----------------------------
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
##ignore patient 6
#npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy') and not f.startswith('p6_')]
print(f"Found {len(npy_files)} .npy files:")
for f in npy_files:
    print(" -", f)

# ----------------------------
# Process each file individually
# ----------------------------
rqa_list = []
patient_dfs = []  # To store per-patient DataFrames

for file_name in npy_files:
    file_path = os.path.join(data_folder, file_name)
    print(f"\nProcessing {file_name} ...")
    
    # Extract patient ID
    match = re.match(r'(p\d+)_', file_name)
    patient_id = match.group(1) if match else "unknown"
    
    data = np.load(file_path)

    #merging of normal segments in segments of 5
    data = merge_normal_segments(data, window_size=10)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    rqa_list.append(data)

    # Extract label (index 16)
    labels = data[:, 0, 0, 16] if data.shape[-1] > 16 else np.zeros(data.shape[0])
    epileptic_indices = np.where(labels == 1)[0]
    epileptic_seconds = epileptic_indices * 2  # 2 sec per row
    print("Epileptic segments (seconds):", epileptic_seconds)

    # Plot per-file features
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, feat_name in enumerate(rqa_feature_names):
        if i >= data.shape[-1]-1:  # skip label column
            break
        means = np.mean(data[:, :, :, i], axis=(1, 2))
        axes[i].plot(means, label='Mean Value')
        axes[i].set_title(feat_name, fontsize=10)
        axes[i].set_xlabel('Sample Number')
        axes[i].set_ylabel('Mean Value')
        axes[i].grid(True)

        # mark epileptic segments
        for idx in epileptic_indices:
            axes[i].axvline(x=idx, color='r', linestyle='--', alpha=0.05,
                            label='Epileptic Segment' if idx == epileptic_indices[0] else "")
        axes[i].legend(fontsize=8)

    for j in range(len(rqa_feature_names), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"EEG Feature Visualization - {file_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plots_folder, f"{file_name}_features.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to: {save_path}")

    # Compute mean across spatial dims for this file
    mean_per_sample = data.mean(axis=(1, 2))
    mean_per_sample = np.nan_to_num(mean_per_sample, nan=0.0, posinf=0.0, neginf=0.0)

    # Split features and label
    features_only = mean_per_sample[:, :-1]  # first 16
    labels = data[:, 0, 0, 16].round().astype(int)

    # Build per-patient DataFrame
    df_patient = pd.DataFrame(features_only, columns=rqa_feature_names)
    df_patient["Label"] = labels
    df_patient["Patient"] = patient_id  # Add patient identifier
    patient_dfs.append(df_patient)

# ----------------------------
# Concatenate all files
# ----------------------------
rqa_all = np.concatenate(rqa_list, axis=0)
print(f"Total shape after concatenation: {rqa_all.shape}")

# Global DataFrame
df_global = pd.concat(patient_dfs, ignore_index=True)

# Compute mean across spatial dims (alternative way, but using pre-computed)
mean_per_sample = rqa_all.mean(axis=(1, 2))
mean_per_sample = np.nan_to_num(mean_per_sample, nan=0.0, posinf=0.0, neginf=0.0)

# Split features and label
features_only = mean_per_sample[:, :-1]  # first 16
labels = rqa_all[:, 0, 0, 16].round().astype(int)

# Build DataFrame (global without patient for original plots)
df = pd.DataFrame(features_only, columns=rqa_feature_names)
df["Label"] = labels

# Print class distribution
normal_count = (df['Label'] == 0).sum()
epileptic_count = (df['Label'] == 1).sum()
total = len(df)
print("------------------------------")
print("Class Distribution:")
print(f"Normal (0): {normal_count} samples ({(normal_count/total)*100:.2f}%)")
print(f"Epileptic (1): {epileptic_count} samples ({(epileptic_count/total)*100:.2f}%)")

# ----------------------------
# Melt for violin plots (global)
# ----------------------------
df_melted = pd.melt(
    df,
    id_vars=['Label'],
    value_vars=rqa_feature_names,
    var_name='Feature',
    value_name='Value'
)

# Split features and last 8 features
first_8_features = rqa_feature_names[:8]
second_8_features = rqa_feature_names[8:]

df_melted_first = df_melted[df_melted['Feature'].isin(first_8_features)]
df_melted_second = df_melted[df_melted['Feature'].isin(second_8_features)]

sns.set_theme(style="whitegrid", palette="colorblind")

# Violin plots first 8 features
g1 = sns.FacetGrid(df_melted_first, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
g1.map(sns.violinplot, 'Label', 'Value', order=[0,1], inner='box', alpha=0.7)
g1.set_titles(col_template="{col_name}")
g1.set_xlabels("Label (0=Normal, 1=Epileptic)")
g1.set_ylabels("Mean Value")
g1.fig.suptitle('RQA Features 1–8: Violin Plots by Label', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
g1.savefig(os.path.join(plots_folder, "violin_features_1-8.png"))
plt.close()

# Violin plots last 8 features
g2 = sns.FacetGrid(df_melted_second, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
g2.map(sns.violinplot, 'Label', 'Value', order=[0,1], inner='box', alpha=0.7)
g2.set_titles(col_template="{col_name}")
g2.set_xlabels("Label (0=Normal, 1=Epileptic)")
g2.set_ylabels("Mean Value")
g2.fig.suptitle('RQA Features 9–16: Violin Plots by Label', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
g2.savefig(os.path.join(plots_folder, "violin_features_9-16.png"))
plt.close()

# ----------------------------
# Per-Patient Boxplots
# ----------------------------
print("\nGenerating per-patient boxplots...")

# Group by patient
patients = df_global['Patient'].unique()
for patient_id in sorted(patients):
    df_patient = df_global[df_global['Patient'] == patient_id]
    
    # Melt for this patient
    df_melted_patient = pd.melt(
        df_patient,
        id_vars=['Label'],
        value_vars=rqa_feature_names,
        var_name='Feature',
        value_name='Value'
    )
    
    # Split into first 8 and second 8 for better visualization
    df_melted_first_p = df_melted_patient[df_melted_patient['Feature'].isin(first_8_features)]
    df_melted_second_p = df_melted_patient[df_melted_patient['Feature'].isin(second_8_features)]
    
    # Boxplots first 8 features
    g1_p = sns.FacetGrid(df_melted_first_p, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
    g1_p.map(sns.boxplot, 'Label', 'Value', order=[0,1], color='lightblue')
    g1_p.set_titles(col_template="{col_name}")
    g1_p.set_xlabels("Label (0=Normal, 1=Epileptic)")
    g1_p.set_ylabels("Mean Value")
    g1_p.fig.suptitle(f'RQA Features 1–8: Box Plots by Label - Patient {patient_id}', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    g1_p.savefig(os.path.join(plots_folder, f"boxplot_features_1-8_patient_{patient_id}.png"))
    plt.close()
    
    # Boxplots last 8 features
    g2_p = sns.FacetGrid(df_melted_second_p, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
    g2_p.map(sns.boxplot, 'Label', 'Value', order=[0,1], color='lightgreen')
    g2_p.set_titles(col_template="{col_name}")
    g2_p.set_xlabels("Label (0=Normal, 1=Epileptic)")
    g2_p.set_ylabels("Mean Value")
    g2_p.fig.suptitle(f'RQA Features 9–16: Box Plots by Label - Patient {patient_id}', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    g2_p.savefig(os.path.join(plots_folder, f"boxplot_features_9-16_patient_{patient_id}.png"))
    plt.close()
    
    print(f"Saved boxplots for Patient {patient_id}")

# ----------------------------
# Per-Patient Scatter Plots: RR vs DET
# ----------------------------
print("\nGenerating per-patient scatter plots (RR vs DET)...")

for patient_id in sorted(patients):
    df_patient = df_global[df_global['Patient'] == patient_id]
    
    # Extract RR and DET
    rr_values = df_patient['Recurrence rate (RR)']
    det_values = df_patient['Determinism (DET)']
    labels = df_patient['Label']
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_patient, x='Recurrence rate (RR)', y='Determinism (DET)', hue='Label', 
                    palette={0: 'blue', 1: 'red'}, s=50, alpha=0.7)
    plt.title(f'Scatter Plot: RR vs DET by Label - Patient {patient_id}')
    plt.xlabel('Recurrence Rate (RR)')
    plt.ylabel('Determinism (DET)')
    plt.legend(title='Label', labels=['Normal (0)', 'Epileptic (1)'])
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(plots_folder, f"scatter_rr_det_patient_{patient_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter plot for Patient {patient_id}")

print(f"All plots saved in: {plots_folder}")
