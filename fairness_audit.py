import os
import urllib.request
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Step 1: Auto-download COMPAS CSV if missing ---
folder_path = os.path.join(os.path.expanduser("~"),
                           r"AppData\\Roaming\\Python\\Python313\\site-packages\\aif360\\data\\raw\\compas")
file_name = "compas-scores-two-years.csv"
full_path = os.path.join(folder_path, file_name)

if not os.path.isfile(full_path):
    os.makedirs(folder_path, exist_ok=True)
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    print(f"Downloading COMPAS CSV from {url} ...")
    urllib.request.urlretrieve(url, full_path)
    print("Download complete!")

# --- Step 2: Load dataset ---
df = pd.read_csv(full_path)

# --- Step 3: Select and clean essential columns ---
feature_cols = ['age', 'sex', 'priors_count', 'c_charge_degree']
essential_cols = feature_cols + ['two_year_recid', 'race']

# Keep only essential columns and drop NA
df = df[essential_cols].dropna().reset_index(drop=True)

# --- Step 4: Convert categorical to numeric ---
df['race'] = df['race'].apply(lambda x: 1 if x == 'Caucasian' else 0)        # 1 = Privileged
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)               # 1 = Male
df['c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x == 'F' else 0)  # 1 = Felony

# --- Step 5: Verify no NA remains ---
if df.isna().sum().sum() > 0:
    raise ValueError("There are still missing values in the dataset!")

# --- Step 6: Create BinaryLabelDataset ---
dataset = BinaryLabelDataset(
    df=df,
    label_names=['two_year_recid'],
    protected_attribute_names=['race'],
    favorable_label=0,
    unfavorable_label=1
)

# --- Step 7: Basic fairness metrics ---
privileged = [{'race': 1}]
unprivileged = [{'race': 0}]

metric = BinaryLabelDatasetMetric(dataset,
                                  unprivileged_groups=unprivileged,
                                  privileged_groups=privileged)

print("=== Basic Metrics ===")
print("Disparate Impact Ratio:", metric.disparate_impact())
print("Mean difference (favorable outcome):", metric.mean_difference())

# --- Step 8: Classification audit ---
X = df[feature_cols].values
y = df['two_year_recid'].values
protected = df['race'].values   # keep protected attribute aligned

# Split with protected attribute included
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected, test_size=0.3, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Build test DataFrame correctly
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df['two_year_recid'] = y_test
test_df['race'] = prot_test   # protected attribute aligned with test set

dataset_test = BinaryLabelDataset(
    df=test_df,
    label_names=['two_year_recid'],
    protected_attribute_names=['race'],
    favorable_label=0,
    unfavorable_label=1
)

# Build predicted DataFrame
pred_df = test_df.copy()
pred_df['two_year_recid'] = y_pred

dataset_pred = BinaryLabelDataset(
    df=pred_df,
    label_names=['two_year_recid'],
    protected_attribute_names=['race'],
    favorable_label=0,
    unfavorable_label=1
)

# Now metrics will align
class_metric = ClassificationMetric(dataset_test, dataset_pred,
                                    unprivileged_groups=unprivileged,
                                    privileged_groups=privileged)

print("\n=== Classification Metrics ===")
print("False Positive Rate difference:", class_metric.false_positive_rate_difference())
print("False Negative Rate difference:", class_metric.false_negative_rate_difference())

# --- Step 9: Visualization ---
# Pass tuple-of-tuples to make them hashable
fpr_priv = class_metric.false_positive_rate((('race', 1),))
fpr_unpriv = class_metric.false_positive_rate((('race', 0),))

plt.bar(["Privileged (Caucasian)", "Unprivileged (African-American)"],
        [fpr_priv, fpr_unpriv], color=['green','red'])
plt.title("False Positive Rate by Race")
plt.ylabel("FPR")
plt.show()

print("\nAudit complete. Metrics and visualization ready.")