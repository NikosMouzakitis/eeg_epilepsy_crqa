import os
import numpy as np
import time
from itertools import product
from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
reduced_folder = os.getcwd()
n_features = 16
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

N_JOBS = 4

# -------------------------------------------------
# PARAMETER GRID (gamma excludes "scale" and "auto")
# -------------------------------------------------
param_grid = {
    "kernel": ["rbf"],
    #"kernel": ["linear", "rbf", "poly", "sigmoid"],
    "C": [100,1000],
    "gamma": [0.25],       # <--- "scale" and "auto" removed
    "degree": [6],
    "coef0": [2.0,],
    "shrinking": [False],
    #"shrinking": [True, False],
    "tol": [1e-4,]
    #"tol": [1e-4, 1e-3]
}

# -------------------------------------------------
# LOAD PATIENT DATA
# -------------------------------------------------
patient_files = [f for f in os.listdir(reduced_folder) if f.endswith('_reduced.npy')]
patient_files.sort()

patient_data = {}
for f in patient_files:
    pid = f.split('_')[0]
    arr = np.load(os.path.join(reduced_folder, f))
    X = arr[:, :n_features]
    y = arr[:, n_features]
    patient_data[pid] = (X, y)

patients = list(patient_data.keys())
print(f"Loaded {len(patients)} patients")

# -------------------------------------------------
# VALIDITY FILTER FOR SVM PARAMETERS
# -------------------------------------------------
def valid_combo(kernel, gamma, degree, coef0):
    """
    This ensures the combination is valid for sklearn's SVC.
    """

    # linear ------------------------------
    if kernel == "linear":
        # invalid if degree or coef0 is used
        if degree is not None:
            return False
        if coef0 is not None:
            return False
        return True

    # rbf --------------------------------
    if kernel == "rbf":
        if degree is not None:
            return False
        if coef0 is not None:
            return False
        return True

    # poly -------------------------------
    if kernel == "poly":
        if degree is None:
            return False
        if coef0 is None:
            return False
        return True

    # sigmoid ----------------------------
    if kernel == "sigmoid":
        if degree is not None:
            return False
        if coef0 is None:
            return False
        return True

    return False

# -------------------------------------------------
# BUILD VALID CONFIGURATIONS
# -------------------------------------------------
all_combos = list(product(
    param_grid["kernel"],
    param_grid["C"],
    param_grid["gamma"],
    param_grid["degree"] + [None],   # allow None, filter later
    param_grid["coef0"] + [None],    # allow None, filter later
    param_grid["shrinking"],
    param_grid["tol"]
))

valid_configs = [
    c for c in all_combos
    if valid_combo(kernel=c[0], gamma=c[2], degree=c[3], coef0=c[4])
]

TOTAL = len(valid_configs)
print(f"Total VALID SVM parameter configurations: {TOTAL}\n")


# -------------------------------------------------
# FUNCTION: RUN ONE CONFIG
# -------------------------------------------------
def run_config(config_index, cfg):
    (kernel, C, gamma, degree, coef0, shrinking, tol) = cfg

    print(f"[{config_index+1}/{TOTAL}] Running: kernel={kernel}, C={C}, gamma={gamma}")

    # Build SVC parameter dict
    params = {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "shrinking": shrinking,
        "tol": tol,
        "class_weight": "balanced"
    }

    if kernel == "poly":
        params["degree"] = degree

    if kernel in ["poly", "sigmoid"]:
        params["coef0"] = coef0

    # Store LOSO metrics
    per_metrics = []

    for test_pid in patients:
        X_train = np.vstack([patient_data[p][0] for p in patients if p != test_pid])
        y_train = np.hstack([patient_data[p][1] for p in patients if p != test_pid])
        X_test, y_test = patient_data[test_pid]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)

        per_metrics.append([acc, sens, spec, f1])

    arr = np.array(per_metrics)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    return {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "shrinking": shrinking,
        "tol": tol,
        "mean_acc": mean[0], "std_acc": std[0],
        "mean_sens": mean[1], "std_sens": std[1],
        "mean_spec": mean[2], "std_spec": std[2],
        "mean_f1": mean[3], "std_f1": std[3]
    }


# -------------------------------------------------
# PARALLEL EXECUTION
# -------------------------------------------------
start = time.time()

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_config)(i, cfg) for i, cfg in enumerate(valid_configs)
)

end = time.time()
print(f"\nFinished ALL runs in {end - start:.2f} seconds")


# -------------------------------------------------
# PRINT TABLE
# -------------------------------------------------
table = []
for r in results:
    table.append([
        r["kernel"], r["C"], r["gamma"], r["degree"], r["coef0"],
        r["shrinking"], r["tol"],
        f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f}",
        f"{r['mean_sens']:.3f} ± {r['std_sens']:.3f}",
        f"{r['mean_spec']:.3f} ± {r['std_spec']:.3f}",
        f"{r['mean_f1']:.3f} ± {r['std_f1']:.3f}",
    ])

print("\n==================== RESULTS ====================")
print(tabulate(
    table,
    headers=[
        "Kernel", "C", "Gamma", "Degree", "Coef0", "Shrink", "Tol",
        "Accuracy", "Sensitivity", "Specificity", "F1"
    ],
    tablefmt="grid"
))

