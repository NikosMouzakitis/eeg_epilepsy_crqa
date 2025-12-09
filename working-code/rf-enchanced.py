import os
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate

# ============================================================
# SETTINGS
# ============================================================

reduced_folder = os.getcwd()     # Folder where patient .npy files are stored
n_features = 16                  # Number of features per sample
random_state = 42

# Each file must contain:
# shape = (n_samples, n_features + 1)
# last column = label (0 or 1)

# ============================================================
# RANDOM FOREST PARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["min_samples_leaf"],
    param_grid["max_features"]
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

# Load all patients into memory
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
# LOSO RANDOM FOREST GRID SEARCH
# ============================================================

results = []

for cfg_id, (n_estimators, max_depth, min_split, min_leaf, max_features) in enumerate(param_combinations, 1):

    print(f"\n[{cfg_id}/{len(param_combinations)}] "
          f"RF | n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_split={min_split}, min_leaf={min_leaf}, "
          f"max_features={max_features}")

    accs, senss, specs, f1s = [], [], [], []

    for test_patient in patients:

        # ----- Split LOSO -----
        X_test, y_test = patients[test_patient]

        X_train, y_train = [], []

        for train_patient in patients:
            if train_patient != test_patient:
                X_train.append(patients[train_patient][0])
                y_train.append(patients[train_patient][1])

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        # ----- Train RF -----
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
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

    # ----- Store mean & std -----
    results.append([
        n_estimators, max_depth, min_split, min_leaf, max_features,
        np.mean(accs), np.std(accs),
        np.mean(senss), np.std(senss),
        np.mean(specs), np.std(specs),
        np.mean(f1s), np.std(f1s)
    ])

# ============================================================
# FINAL RESULTS TABLE
# ============================================================

headers = [
    "Trees", "MaxDepth", "MinSplit", "MinLeaf", "MaxFeat",
    "Mean Acc", "Std",
    "Mean Sens", "Std",
    "Mean Spec", "Std",
    "Mean F1", "Std"
]

print("\n" + "="*120)
print(tabulate(results, headers=headers, floatfmt=".4f"))
print("="*120)

