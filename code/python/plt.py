import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the matrices
# -----------------------------
file1 = "p24_04_epileptic512.npy"  
file2="p24_1_epileptic_512.npy"                                                                                                
file3 = "p24_3_1_normal512.npy"                                                                                                  
file4 = "p24_3_3_normal512.npy"
file5= "p24_3_epileptic_256.npy"
file6 = "p24_1_3_normal_512.npy"
file7 = "p24_04_1_normal512.npy"
file8="p24_06_epileptic512.npy"
file9="p24_06_normal512.npy"
rqa1 = np.load(file1)  # shape: [samples1, 22, 22, 17]
rqa2 = np.load(file2)  # shape: [samples2, 22, 22, 17]
rqa3 = np.load(file3)  # shape: [samples2, 22, 22, 17]
rqa4 = np.load(file4)  # shape: [samples2, 22, 22, 17]
rqa5 = np.load(file5)  # shape: [samples2, 22, 22, 17]
rqa6 = np.load(file6)  # shape: [samples2, 22, 22, 17]
rqa7 = np.load(file7)  # shape: [samples2, 22, 22, 17]
rqa8 = np.load(file8)  # shape: [samples2, 22, 22, 17]
rqa9 = np.load(file9)  # shape: [samples2, 22, 22, 17]

print("------------------------------_")
print(rqa1.shape)
print(rqa2.shape)
print(rqa3.shape)
print(rqa4.shape)
print(rqa5.shape)
print(rqa6.shape)
print(rqa7.shape)
print(rqa8.shape)
print(rqa9.shape)
# -----------------------------
# 2. Concatenate along first dimension
# -----------------------------
rqa_all = np.concatenate([rqa1, rqa2,rqa3,rqa4,rqa5,rqa6,rqa7,rqa8,rqa9], axis=0)  # shape: [samples1+samples2, 22, 22, 17]
rqa_all = np.load("p1_31_512.npy")

print("------------------------------_")
print("total shape")
print(rqa_all.shape)
# -----------------------------
# 3. Flatten electrode dimensions: average across channel pairs
# -----------------------------
# Result: [samples, 17]
mean_per_sample = rqa_all.mean(axis=(1,2))
print("Mean matrix")
print(mean_per_sample.shape)


# -----------------------------
# 4. Create DataFrame for plotting
# -----------------------------
feature_names = [f"Feature {i+1}" for i in range(16)] + ["Label"]
df = pd.DataFrame(mean_per_sample, columns=feature_names)

# Ensure label is integer
df["Label"] = df["Label"].round().astype(int)

# -----------------------------
# 5. Plot separate boxplots per feature
# -----------------------------
plt.figure(figsize=(20, 12))

for i, fname in enumerate(feature_names[:-1], 1):  # exclude "Label"
    plt.subplot(4, 4, i)  # 4x4 grid
    sns.boxplot(x="Label", y=fname, data=df)
    plt.title(fname)
    plt.xlabel("Label (0=Normal, 1=Epileptic)")
    plt.ylabel("Mean Value")

plt.tight_layout()
plt.show()




import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your mean_matrix
mean_matrix = mean_per_sample

# Split features and labels
X = mean_matrix[:, :-1]  # Features (1655, 16)
y = mean_matrix[:, -1]   # Labels (1655,)

# Standardize features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize SVM with class weighting to handle imbalance
svm_classifier = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Epileptic']))

# Cross-validation to assess robustness
cv_scores = cross_val_score(svm_classifier, X, y, cv=5, scoring='f1_macro')
print(f"\nCross-Validation F1 Macro Scores: {cv_scores}")
print(f"Mean CV F1 Macro Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Optional: Visualize confusion matrix or ROC curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Epileptic'], yticklabels=['Normal', 'Epileptic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve (if probability estimates are needed)
y_prob = svm_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
