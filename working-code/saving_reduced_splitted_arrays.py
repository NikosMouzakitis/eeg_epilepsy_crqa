import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------
# Settings
# ----------------------------
data_folder = os.getcwd()
plots_folder = os.path.join(data_folder, "plots")
global_folder = os.path.join(plots_folder, "global")
patients_folder = os.path.join(plots_folder, "patients")
reduced_folder = os.path.join(data_folder, "reduced")

os.makedirs(global_folder, exist_ok=True)
os.makedirs(patients_folder, exist_ok=True)
os.makedirs(reduced_folder, exist_ok=True)

rqa_feature_names = [
    "Recurrence rate (RR)", "Determinism (DET)", "Average diagonal line length (L)",
    "Longest diagonal line length (L_max)", "Divergence (DIV)", "Entropy diagonal lines (L_entr)",
    "Laminarity (LAM)", "Trapping time (TT)", "Longest vertical line length (V_max)",
    "Entropy vertical lines (V_entr)", "Average white vertical line length (W)",
    "Longest white vertical line length (W_max)", "Longest white vertical line length divergence (W_div)",
    "Entropy white vertical lines (W_entr)", "DET/RR", "LAM/DET"
]

# ----------------------------
# Load .npy files
# ----------------------------
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
print(f"Found {len(npy_files)} .npy files:")
for f in npy_files: print(" -", f)

# ----------------------------
# Process each recording separately
# ----------------------------
all_dfs = []
per_patient_dfs = {}

for file_name in npy_files:
    file_path = os.path.join(data_folder, file_name)
    match = re.match(r'(p\d+)_', file_name)
    patient_id = match.group(1) if match else "unknown"

    data = np.load(file_path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mean_per_sample = data.mean(axis=(1, 2))
    features_only = mean_per_sample[:, :-1]
    labels = data[:, 0, 0, 16].round().astype(int)

    df_patient = pd.DataFrame(features_only, columns=rqa_feature_names)
    df_patient["Label"] = labels
    df_patient["Patient"] = patient_id

    # Separate normals and epileptics
    df_norm = df_patient[df_patient["Label"] == 0].copy()
    df_epil = df_patient[df_patient["Label"] == 1].copy()

    # Creation of 10 averaged normal vectors per recording
    if not df_norm.empty:
        n_samples = df_norm.shape[0]
        n_splits = 40 
        split_idx = np.array_split(np.arange(n_samples), n_splits)
        avg_normals_list = []
        for i, idx in enumerate(split_idx):
            avg_vector = df_norm.iloc[idx][rqa_feature_names].mean(axis=0)
            avg_normals_list.append(avg_vector)
        df_norm_avg = pd.DataFrame(avg_normals_list, columns=rqa_feature_names)
        df_norm_avg["Label"] = 0
        df_norm_avg["Patient"] = [f"{patient_id}_avgNorm{i+1}" for i in range(len(avg_normals_list))]
    else:
        df_norm_avg = pd.DataFrame(columns=rqa_feature_names + ["Label","Patient"])

    # Combine averaged normals with epileptics
    df_combined = pd.concat([df_norm_avg, df_epil], ignore_index=True)
    all_dfs.append(df_combined)

    # Store per-patient DataFrame for plotting and saving
    if patient_id not in per_patient_dfs:
        per_patient_dfs[patient_id] = []
    per_patient_dfs[patient_id].append(df_combined)

# ----------------------------
# Global DataFrame
# ----------------------------
df_global = pd.concat(all_dfs, ignore_index=True)
print(f"Total samples: {len(df_global)}")

# ----------------------------
# Print class distribution
# ----------------------------
normal_count = (df_global['Label'] == 0).sum()
epileptic_count = (df_global['Label'] == 1).sum()
total = len(df_global)
print("------------------------------")
print("Class Distribution:")
print(f"Normal (0): {normal_count} samples ({(normal_count/total)*100:.2f}%)")
print(f"Epileptic (1): {epileptic_count} samples ({(epileptic_count/total)*100:.2f}%)")

# ----------------------------
# Plotting helpers
# ----------------------------
def violin_plot(df, save_path, title):
    df_melted = pd.melt(df, id_vars=['Label'], value_vars=rqa_feature_names,
                        var_name='Feature', value_name='Value')
    df_melted['Label_str'] = df_melted['Label'].astype(str)
    first_8_features = rqa_feature_names[:8]
    second_8_features = rqa_feature_names[8:]

    # First 8
    df_first = df_melted[df_melted['Feature'].isin(first_8_features)]
    g1 = sns.FacetGrid(df_first, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
    g1.map_dataframe(sns.violinplot, x='Label_str', y='Value', order=['0','1'], inner='box',
                     palette={'0':'lightblue','1':'salmon'})
    g1.set_titles(col_template="{col_name}")
    g1.set_xlabels("Label (0=Normal, 1=Epileptic)")
    g1.set_ylabels("Value")
    g1.fig.suptitle(title + " (Features 1-8)", fontsize=16, y=1.02)
    plt.tight_layout()
    g1.savefig(save_path.replace(".png","_1-8.png"))
    plt.close()

    # Last 8
    df_second = df_melted[df_melted['Feature'].isin(second_8_features)]
    g2 = sns.FacetGrid(df_second, col='Feature', col_wrap=4, height=3, aspect=1.2, sharey=False)
    g2.map_dataframe(sns.violinplot, x='Label_str', y='Value', order=['0','1'], inner='box',
                     palette={'0':'lightblue','1':'salmon'})
    g2.set_titles(col_template="{col_name}")
    g2.set_xlabels("Label (0=Normal, 1=Epileptic)")
    g2.set_ylabels("Value")
    g2.fig.suptitle(title + " (Features 9-16)", fontsize=16, y=1.02)
    plt.tight_layout()
    g2.savefig(save_path.replace(".png","_9-16.png"))
    plt.close()

def scatter_plot(df, save_path, title):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='Recurrence rate (RR)', y='Determinism (DET)',
                    hue='Label', palette={0:'blue',1:'red'}, s=80, alpha=0.7)
    plt.title(title)
    plt.xlabel('Recurrence Rate (RR)')
    plt.ylabel('Determinism (DET)')
    plt.legend(title='Label', labels=['Normal (0)', 'Epileptic (1)'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# Global plots
# ----------------------------
violin_plot(df_global, os.path.join(global_folder,"global_violin.png"), "Global RQA Features")
scatter_plot(df_global, os.path.join(global_folder,"global_scatter_rr_det.png"), "Global RR vs DET Scatter")

# ----------------------------
# Per-patient plots, print shapes & save reduced vectors
# ----------------------------
for patient_id, dfs_list in per_patient_dfs.items():
    df_patient = pd.concat(dfs_list, ignore_index=True)
    
    # Count samples
    total_samples = len(df_patient)
    normal_samples = (df_patient['Label'] == 0).sum()
    epileptic_samples = (df_patient['Label'] == 1).sum()
    
    # Print shapes
    print(f"Patient {patient_id}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Normal samples: {normal_samples}")
    print(f"  Epileptic samples: {epileptic_samples}")
    print(f"  Reduced vector shape: {df_patient[rqa_feature_names + ['Label']].to_numpy().shape}\n")
    
    # Save reduced vectors as .npy
    reduced_vectors = df_patient[rqa_feature_names + ["Label"]].to_numpy()
    np.save(os.path.join(reduced_folder, f"{patient_id}_reduced.npy"), reduced_vectors)
    
    # Plots
    violin_plot(df_patient, os.path.join(patients_folder,f"{patient_id}_violin.png"), f"Patient {patient_id} RQA Features")
    scatter_plot(df_patient, os.path.join(patients_folder,f"{patient_id}_scatter_rr_det.png"), f"Patient {patient_id} RR vs DET Scatter")

print(f"All plots saved in: {plots_folder}")
print(f"All reduced vectors saved in: {reduced_folder}")

