import os
import numpy as np
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# ============================================================
# SETTINGS
# ============================================================

reduced_folder = os.getcwd()   # Folder with patient .npy files
n_features = 16                # Number of features per sample

# Each file must contain:
# shape = (n_samples, n_features + 1)
# last column = label (0 or 1)

# ============================================================
# kNN PARAMETER GRID
# ============================================================

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree"]
}

param_combinations = list(product(
    param_grid["n_neighbors"],
    param_grid["metric"],
    param_grid["weights"],
    param_grid["algorithm"]
))

# ============================================================
# LOAD PATIENT FILES
# ============================================================

patient_files = sorted([
    f for f in os.listdir(reduced_folder)
    if f.endswith(".npy")
])

if not patient_files:
    raise RuntimeError("❌ No .npy patient files found!")

print(f"✅ Loaded {len(patient_files)} patients")

patients = {}

for file in patient_files:
    path = os.path.join(reduced_folder, file)
    data = np.load(path)

    if data.ndim != 2 or data.shape[1] != n_features + 1:
        raise ValueError(f"❌ Invalid shape in {file}: {data.shape}")

    X = data[:, :n_features]
    y = data[:, n_features].astype(int)

    if len(np.unique(y)) < 2:
        print(f"⚠️ Warning: {file} contains only one class")

    patients[file] = (X, y)

# ============================================================
# METRIC HELPERS
# ============================================================

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, sens, spec, f1

# ============================================================
# LOSO kNN GRID SEARCH
# ============================================================

results = []

for cfg_id, (k, metric, weights, algorithm) in enumerate(param_combinations, 1):

    print(f"\n[{cfg_id}/{len(param_combinations)}] "
          f"kNN | k={k}, metric={metric}, weights={weights}, algorithm={algorithm}")

    accs, senss, specs, f1s = [], [], [], []

    for test_patient in patients:

        # ----- LOSO SPLIT -----
        X_test, y_test = patients[test_patient]

        X_train, y_train = [], []

        for train_patient in patients:
            if train_patient != test_patient:
                X_train.append(patients[train_patient][0])
                y_train.append(patients[train_patient][1])

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        # ----- STANDARDIZATION (CRITICAL FOR kNN) -----
        scaler = StandardScaler()
        #X_train = scaler.fit_transform(X_train)
        #X_test = scaler.transform(X_test)

        # ----- Train kNN -----
        clf = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            algorithm=algorithm
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # ----- Metrics -----
        acc, sens, spec, f1 = compute_metrics(y_test, y_pred)

        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        f1s.append(f1)

        print(f"  Patient {test_patient.replace('.npy','')}: "
              f"Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, F1={f1:.3f}")

    # ----- Store Mean & Std -----
    results.append([
        k, metric, weights, algorithm,
        np.mean(accs), np.std(accs),
        np.mean(senss), np.std(senss),
        np.mean(specs), np.std(specs),
        np.mean(f1s), np.std(f1s)
    ])

# ============================================================
# FINAL RESULTS TABLE (MATCHES RF FORMAT)
# ============================================================

headers = [
    "k", "Metric", "Weights", "Algorithm",
    "Mean Acc", "Std",
    "Mean Sens", "Std",
    "Mean Spec", "Std",
    "Mean F1", "Std"
]

print("\n" + "=" * 120)
print(tabulate(results, headers=headers, floatfmt=".4f"))
print("=" * 120)

