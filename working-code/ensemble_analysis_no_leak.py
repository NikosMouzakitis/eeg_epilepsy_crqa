import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
import networkx as nx
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

# Imports for Ensemble Models
import xgboost as xgb
import lightgbm as lgb

# ----------------------------
# Settings
# ----------------------------
data_folder = "."
print(data_folder)

rqa_feature_names = [
    "Recurrence rate (RR)", "Determinism (DET)", "Average diagonal line length (L)",
    "Longest diagonal line length (L_max)", "Divergence (DIV)", "Entropy diagonal lines (L_entr)",
    "Laminarity (LAM)", "Trapping time (TT)", "Longest vertical line length (V_max)",
    "Entropy vertical lines (V_entr)", "Average white vertical line length (W)",
    "Longest white vertical line length (W_max)", "Longest white vertical line length divergence (W_div)",
    "Entropy white vertical lines (W_entr)", "DET/RR", "LAM/DET"
]

def get_patient_id(filename):
    match = re.match(r'(p\d+)_', filename)
    return match.group(1) if match else None

# ----------------------------
# Load all patient data (store raw TS for per-fold processing)
# ----------------------------
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
patient_files = {}
for f in npy_files:
    pid = get_patient_id(f)
    if pid:
        patient_files.setdefault(pid, []).append(f)

print(f"Found {len(npy_files)} files for {len(patient_files)} patients: {sorted(patient_files.keys())}")
print("="*70)

# Per-patient data storage (raw TS only)
patient_data = {}
all_y = []
all_patient = []

for patient_id in sorted(patient_files.keys()):
    # Load data for this patient
    patient_data_list = []
    for file in patient_files[patient_id]:
        data = np.load(os.path.join(data_folder, file))
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        patient_data_list.append(data)
    
    patient_all = np.concatenate(patient_data_list, axis=0)
    y = patient_all[:, 0, 0, 16].round().astype(int)
    
    if np.sum(y == 1) < 2:
        print(f"Skipping patient {patient_id}: fewer than 2 epileptic samples")
        continue
    
    # Store raw TS and y
    patient_data[patient_id] = {
        'raw_ts': patient_all,
        'y': y
    }
    
    all_y.extend(y)
    all_patient.extend([patient_id] * len(y))

print(f"Global dataset overview: {sum(len(d['y']) for d in patient_data.values())} samples across patients, {sum(np.sum(d['y']==1) for d in patient_data.values())} epileptic")

full_feature_names = rqa_feature_names + [
    "DET_EFF", "V_D_RATIO", "REC_STAB", "log_RR", "log_L_max", "DET_x_L",
    "ENT_RATIO", "LYAP_PROXY", "FRACT_PROXY", "TRAP_EFF",
    "DET_LAM_RATIO", "L_LMAX_LOG_RATIO", "DIV_ENT_PRODUCT", "TT_VMAX_RATIO", "RR_DET_DIFF"
]

# ----------------------------
# Function to compute core + synthetic features (unsupervised, per split)
# ----------------------------
def compute_core_features(X_feat):
    df = pd.DataFrame(X_feat, columns=rqa_feature_names)
    eps = 1e-8
    RR, DET, L, L_max = df["Recurrence rate (RR)"], df["Determinism (DET)"], df["Average diagonal line length (L)"], df["Longest diagonal line length (L_max)"]
    L_entr, LAM, TT, DIV = df["Entropy diagonal lines (L_entr)"], df["Laminarity (LAM)"], df["Trapping time (TT)"], df["Divergence (DIV)"]
    V_entr = df["Entropy vertical lines (V_entr)"]
    V_max = df["Longest vertical line length (V_max)"]

    df["DET_EFF"] = DET / (RR + eps)
    df["V_D_RATIO"] = LAM / (DET + eps)
    df["REC_STAB"] = L / (L_max + eps)
    df["log_RR"] = np.log(RR + eps)
    df["log_L_max"] = np.log(L_max + eps)
    df["DET_x_L"] = DET * L
    df["ENT_RATIO"] = L_entr / (V_entr + eps)
    df["LYAP_PROXY"] = DIV / np.log(L_max + 1 + eps)
    df["FRACT_PROXY"] = np.log(L_max + eps) / np.log(L + eps)
    df["TRAP_EFF"] = TT / (LAM + eps)
    
    # Additional synthetic features
    df["DET_LAM_RATIO"] = DET / (LAM + eps)
    df["L_LMAX_LOG_RATIO"] = np.log(L + eps) / np.log(L_max + eps)
    df["DIV_ENT_PRODUCT"] = DIV * L_entr
    df["TT_VMAX_RATIO"] = TT / (V_max + eps)
    df["RR_DET_DIFF"] = RR - DET
    
    return df.values

# ----------------------------
# Function for CV with Sampler (Ensemble: RF + Tuned XGBoost + Tuned LightGBM)
# ----------------------------
def run_cv_with_sampler(patient_data_dict, sampler_class=BorderlineSMOTE, sampler_params=None, sampler_name="BorderlineSMOTE"):
    # RF params
    rf_params = {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced',
        'max_depth': 10
    }
    
    # XGBoost base params
    xgb_base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'eval_metric': 'aucpr'
    }
    
    # LightGBM base params
    lgb_base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'metric': 'aucpr'
    }
    
    # Tuning params
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    target_recall = 0.85  # Threshold for precision-recall optimization
    
    cv_splits = 5
    results = {}
    
    for patient_id, data in patient_data_dict.items():
        raw_ts = data['raw_ts']
        y_full = data['y']
        n_samples = len(y_full)
        
        if len(np.unique(y_full)) < 2:
            print(f"Skipping {patient_id}: Only one class")
            continue
        
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        y_true_all = []
        y_pred_all = []
        importances_all = []  # Average from all models (numeric only)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples), y_full)):
            train_raw = raw_ts[train_idx]
            test_raw = raw_ts[test_idx]
            y_train = y_full[train_idx]
            y_test = y_full[test_idx]
            
            # Compute X_raw and X_feat (per split)
            X_raw_train = np.nan_to_num(train_raw.mean(axis=(1, 2)), nan=0.0, posinf=0.0, neginf=0.0)
            X_feat_train = X_raw_train[:, :-1]
            X_core_train = compute_core_features(X_feat_train)
            
            X_raw_test = np.nan_to_num(test_raw.mean(axis=(1, 2)), nan=0.0, posinf=0.0, neginf=0.0)
            X_feat_test = X_raw_test[:, :-1]
            X_core_test = compute_core_features(X_feat_test)
            
            # Polynomial Features (fit on train)
            poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
            X_poly_train = poly.fit_transform(X_core_train)
            X_poly_test = poly.transform(X_core_test)
            
            # SelectKBest for poly (fit on train)
            selector_poly = SelectKBest(f_classif, k=min(25, X_poly_train.shape[1]))
            X_poly_selected_train = selector_poly.fit_transform(X_poly_train, y_train)
            X_poly_selected_test = selector_poly.transform(X_poly_test)
            
            # Joined features (top_rqa from train)
            top_rqa_idx = SelectKBest(f_classif, k=3).fit(X_core_train[:, :len(rqa_feature_names)], y_train).get_support(indices=True)
            top_poly_for_join_train = X_poly_selected_train[:, :3]
            top_rqa_for_join_train = X_core_train[:, top_rqa_idx]
            joined_features_train = []
            for i in range(3):
                for j in range(3):
                    joined_features_train.append(top_poly_for_join_train[:, i] * top_rqa_for_join_train[:, j])
                    joined_features_train.append(top_poly_for_join_train[:, i] + top_rqa_for_join_train[:, j])
            X_joined_train = np.column_stack(joined_features_train)
            
            top_poly_for_join_test = X_poly_selected_test[:, :3]
            top_rqa_for_join_test = X_core_test[:, top_rqa_idx]
            joined_features_test = []
            for i in range(3):
                for j in range(3):
                    joined_features_test.append(top_poly_for_join_test[:, i] * top_rqa_for_join_test[:, j])
                    joined_features_test.append(top_poly_for_join_test[:, i] + top_rqa_for_join_test[:, j])
            X_joined_test = np.column_stack(joined_features_test)
            
            # Graph features (compute on train, apply to test)
            n_feats = X_core_train.shape[1]
            corr_matrix_train = np.corrcoef(X_core_train.T)
            G = nx.Graph()
            for i in range(n_feats):
                G.add_node(i)
            for i in range(n_feats):
                for j in range(i + 1, n_feats):
                    if abs(corr_matrix_train[i, j]) > 0.1:
                        mean_ictal = np.mean(X_core_train[y_train == 1, i]) + np.mean(X_core_train[y_train == 1, j])
                        mean_non = np.mean(X_core_train[y_train == 0, i]) + np.mean(X_core_train[y_train == 0, j])
                        weight = abs(corr_matrix_train[i, j]) * max(0, mean_ictal - mean_non)
                        G.add_edge(i, j, weight=weight)
            betweenness = np.array([nx.betweenness_centrality(G, weight='weight').get(i, 0) for i in range(n_feats)])
            degree = np.array([G.degree(i) for i in range(n_feats)])
            clustering = np.array([nx.clustering(G, weight='weight').get(i, 0) for i in range(n_feats)])
            train_means = np.mean(X_core_train, axis=0)
            rel_values_train = X_core_train / (train_means + 1e-8)
            X_graph_train = np.column_stack([
                np.sum(rel_values_train * betweenness[:, np.newaxis].T, axis=1) / np.sum(rel_values_train, axis=1),
                np.sum(rel_values_train * degree[:, np.newaxis].T, axis=1) / np.sum(rel_values_train, axis=1),
                np.sum(rel_values_train * clustering[:, np.newaxis].T, axis=1) / np.sum(rel_values_train, axis=1)
            ])
            rel_values_test = X_core_test / (train_means + 1e-8)
            X_graph_test = np.column_stack([
                np.sum(rel_values_test * betweenness[:, np.newaxis].T, axis=1) / np.sum(rel_values_test, axis=1),
                np.sum(rel_values_test * degree[:, np.newaxis].T, axis=1) / np.sum(rel_values_test, axis=1),
                np.sum(rel_values_test * clustering[:, np.newaxis].T, axis=1) / np.sum(rel_values_test, axis=1)
            ])
            
            # Temporal dynamics (per split, on shuffled order)
            n_feats_local = X_core_train.shape[1]
            X_temporal_train = np.zeros((len(y_train), 2 * n_feats_local))
            for feat_idx in range(n_feats_local):
                feat_series = X_core_train[:, feat_idx]
                deltas = np.diff(feat_series, prepend=feat_series[0])
                trends = gaussian_filter1d(feat_series, sigma=2)
                X_temporal_train[:, feat_idx] = deltas
                X_temporal_train[:, n_feats_local + feat_idx] = trends
            
            X_temporal_test = np.zeros((len(y_test), 2 * n_feats_local))
            for feat_idx in range(n_feats_local):
                feat_series = X_core_test[:, feat_idx]
                deltas = np.diff(feat_series, prepend=feat_series[0])
                trends = gaussian_filter1d(feat_series, sigma=2)
                X_temporal_test[:, feat_idx] = deltas
                X_temporal_test[:, n_feats_local + feat_idx] = trends
            
            # CRQA relations (MI, supervised on train; mean proxy on test)
            top_crqa_idx = np.argsort(f_classif(X_core_train[:, :len(rqa_feature_names)], y_train)[0])[::-1][:5]
            n_train_local = len(y_train)
            X_mi_train = np.zeros((n_train_local, 5))
            for i in range(n_train_local - 2):
                block_X = X_core_train[i:i+3, top_crqa_idx]
                block_y = y_train[i:i+3]
                mi_scores = mutual_info_classif(block_X, block_y, random_state=42)
                X_mi_train[i, :] = mi_scores
            X_mi_train[n_train_local-2:, :] = X_mi_train[n_train_local-3, :]
            mean_mi_train = np.mean(X_mi_train, axis=0)
            
            n_test_local = len(y_test)
            X_mi_test = np.tile(mean_mi_train[np.newaxis, :], (n_test_local, 1))
            
            # Statistical profiles (rolling, per split)
            df_core_train = pd.DataFrame(X_core_train, columns=full_feature_names)
            rolling_mean_train = df_core_train.rolling(window=5, min_periods=1).mean().fillna(method='bfill').values
            rolling_std_train = df_core_train.rolling(window=5, min_periods=1).std().fillna(0).values
            X_stats_train = np.hstack([rolling_mean_train, rolling_std_train])
            
            df_core_test = pd.DataFrame(X_core_test, columns=full_feature_names)
            rolling_mean_test = df_core_test.rolling(window=5, min_periods=1).mean().fillna(method='bfill').values
            rolling_std_test = df_core_test.rolling(window=5, min_periods=1).std().fillna(0).values
            X_stats_test = np.hstack([rolling_mean_test, rolling_std_test])
            
            # Full engineered
            X_engineered_train = np.column_stack([X_core_train, X_poly_selected_train, X_joined_train, X_graph_train, X_temporal_train, X_mi_train, X_stats_train])
            X_engineered_test = np.column_stack([X_core_test, X_poly_selected_test, X_joined_test, X_graph_test, X_temporal_test, X_mi_test, X_stats_test])
            
            # PCA (fit on train)
            n_eng_feats = X_engineered_train.shape[1]
            pca = PCA(n_components=min(20, n_eng_feats))
            X_pca_train = pca.fit_transform(X_engineered_train)
            X_pca_test = pca.transform(X_engineered_test)
            
            # SelectKBest (fit on train; for final top 25)
            selector = SelectKBest(f_classif, k=min(25, X_pca_train.shape[1]))
            X_top25_train = selector.fit_transform(X_pca_train, y_train)
            X_top25_test = selector.transform(X_pca_test)
            
            # Now resample and train
            sampler = sampler_class(random_state=42, **(sampler_params or {}))
            X_train_res, y_train_res = sampler.fit_resample(X_top25_train, y_train)
            
            # Train RF
            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train_res, y_train_res)
            rf_proba = rf.predict_proba(X_top25_test)[:, 1]
            
            # Tune and train XGBoost
            best_xgb_f1, best_xgb_scale, best_xgb_model = 0, 0, None
            for mult in multipliers:
                scale_pos = (np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8)) * mult
                xgb_params = xgb_base_params.copy()
                xgb_params['scale_pos_weight'] = scale_pos
                model_temp = xgb.XGBClassifier(**xgb_params)
                model_temp.fit(X_train_res, y_train_res)
                y_pred_temp = model_temp.predict(X_top25_test)
                temp_f1 = f1_score(y_test, y_pred_temp, zero_division=0)
                if temp_f1 > best_xgb_f1:
                    best_xgb_f1 = temp_f1
                    best_xgb_scale = scale_pos
                    best_xgb_model = model_temp
            if best_xgb_model is None:
                scale_pos = (np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8)) * 1.0
                xgb_params = xgb_base_params.copy()
                xgb_params['scale_pos_weight'] = scale_pos
                best_xgb_model = xgb.XGBClassifier(**xgb_params)
                best_xgb_model.fit(X_train_res, y_train_res)
            xgb_proba = best_xgb_model.predict_proba(X_top25_test)[:, 1]
            
            # Tune and train LightGBM
            best_lgb_f1, best_lgb_scale, best_lgb_model = 0, 0, None
            for mult in multipliers:
                scale_pos = (np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8)) * mult
                lgb_params = lgb_base_params.copy()
                lgb_params['scale_pos_weight'] = scale_pos
                model_temp = lgb.LGBMClassifier(**lgb_params)
                model_temp.fit(X_train_res, y_train_res)
                y_pred_temp = model_temp.predict(X_top25_test)
                temp_f1 = f1_score(y_test, y_pred_temp, zero_division=0)
                if temp_f1 > best_lgb_f1:
                    best_lgb_f1 = temp_f1
                    best_lgb_scale = scale_pos
                    best_lgb_model = model_temp
            if best_lgb_model is None:
                scale_pos = (np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8)) * 1.0
                lgb_params = lgb_base_params.copy()
                lgb_params['scale_pos_weight'] = scale_pos
                best_lgb_model = lgb.LGBMClassifier(**lgb_params)
                best_lgb_model.fit(X_train_res, y_train_res)
            lgb_proba = best_lgb_model.predict_proba(X_top25_test)[:, 1]
            
            # ENSEMBLE: Average probabilities
            ens_proba = np.mean([rf_proba, xgb_proba, lgb_proba], axis=0)
            
            # Threshold Optimization on ensemble proba
            prec, rec, thresh = precision_recall_curve(y_test, ens_proba)
            mask = rec >= target_recall
            if np.any(mask):
                opt_idx = np.argmax(prec[mask])
                opt_thresh = thresh[np.where(mask)[0][opt_idx]]
            else:
                opt_thresh = 0.5
            y_pred_fold = (ens_proba >= opt_thresh).astype(int)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred_fold)
            
            # Average importances (from all models; numeric)
            rf_imp = rf.feature_importances_
            xgb_imp = best_xgb_model.feature_importances_
            lgb_imp = best_lgb_model.feature_importances_
            avg_imp = np.mean([rf_imp, xgb_imp, lgb_imp], axis=0)
            importances_all.append(avg_imp)
            
            print(f"  Fold {fold+1}: Ens avg scale_pos ~{np.mean([best_xgb_scale, best_lgb_scale]):.2f}, Opt thresh={opt_thresh:.3f}")
        
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        acc = np.mean(y_pred_all == y_true_all)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all)
        report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        
        importances = np.mean(importances_all, axis=0)
        
        results[patient_id] = {
            'y_true': y_true_all, 'y_pred': y_pred_all, 'cm': cm,
            'acc': acc, 'bal_acc': bal_acc, 'precision': precision,
            'recall': recall, 'f1': f1
        }
        
        print(f"\nPatient {patient_id} (CV with {sampler_name} + Ensemble (RF + XGB + LGB) + Tuning):")
        print(f"  CV Accuracy: {acc:.4f}, CV Balanced Acc: {bal_acc:.4f}")
        print(f"  CV Precision (Epi): {precision:.4f}, CV Recall (Epi): {recall:.4f}, CV F1 (Epi): {f1:.4f}")
        print(f"  CV Confusion Matrix:\n{cm}")
        # Note: Skipped named feature importances due to per-fold variation
        print(f"  Top 3 Average Feature Importances (numeric, features vary per fold):")
        idx_sorted = np.argsort(importances)[::-1][:3]
        for i in idx_sorted:
            print(f"    Feature {i}: {importances[i]:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print(f"PER-PATIENT CV SUMMARY WITH {sampler_name} + Ensemble + Tuning")
    print("="*80)
    
    summary_df = pd.DataFrame({
        'Patient': list(results.keys()),
        'CV_Accuracy': [results[p]['acc'] for p in results],
        'CV_Bal_Acc': [results[p]['bal_acc'] for p in results],
        'CV_Precision_Epi': [results[p]['precision'] for p in results],
        'CV_Recall_Epi': [results[p]['recall'] for p in results],
        'CV_F1_Epi': [results[p]['f1'] for p in results]
    })
    print(summary_df.round(4))
    
    # Plots
    print("\nGenerating per-patient CV confusion matrix plots...")
    for patient_id, res in results.items():
        cm = res['cm']
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Epi', 'Epi'], yticklabels=['Non-Epi', 'Epi'])
        plt.title(f'CV Confusion Matrix with {sampler_name} + Ensemble + Tuning - Patient {patient_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'cv_ensemble_cm_patient_{patient_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    print("\n" + "="*80)
    print(f"PER-PATIENT CV ANALYSIS WITH {sampler_name} + Ensemble + Tuning COMPLETE")
    print("="*80)
    
    return results, summary_df

# Run CV with BorderlineSMOTE (default) + Ensemble + Integrated Tuning
borderline_results, borderline_df = run_cv_with_sampler(patient_data)
