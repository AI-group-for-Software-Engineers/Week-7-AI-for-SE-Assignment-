# scripts/write_case_studies.py
"""
Writes the Part 2 written case analyses (Markdown file) to outputs/case_studies.md

Usage:
    python scripts/write_case_studies.py --outdir outputs
"""
import os
import argparse

CONTENT = """# AI Ethics — Part 2: Case Study Analyses

## Case 1 — Biased Hiring Tool (Amazon-style)
**Scenario summary:** An AI recruiting tool penalizes female candidates.

### 1. Identification of bias sources
- **Training data bias:** Historical hiring data reflected a workforce skew that favored men.
- **Label bias:** Using past hiring decisions as ground truth replicates human biases.
- **Proxy features:** Features such as names, career-gap indicators, or correlated signals may indirectly encode gender.
- **Objective mismatch:** Optimizing purely for historical hires without fairness constraints.

### 2. Proposed fixes (three)
1. **Preprocessing — Reweighting / Balanced sampling.** Adjust or resample training data to correct imbalances so underrepresented groups are not underlearned.
2. **In-processing — Fairness-aware training.** Use fairness constraints or regularizers (e.g., adding penalty terms for demographic parity or equalized odds).
3. **Post-processing — Threshold adjustment / calibration.** Apply group-specific thresholds or calibration to equalize error rates or selection rates.

Also: remove obvious proxies (names), require human-in-the-loop for final decisions, and log model decisions.

### 3. Metrics to evaluate fairness post-correction
- **Performance:** Accuracy, Precision, Recall, AUC.
- **Group fairness:** Demographic parity difference and disparate impact (ratio).
- **Error parity:** Equalized odds differences (TPR and FPR diffs) and False Negative Rate parity.
- **Calibration:** Calibration curves per group and reliability plots.
- **Statistical tests:** Bootstrap confidence intervals for group differences.

---

## Case 2 — Facial Recognition in Policing

**Scenario summary:** A facial recognition system misidentifies minorities at higher rates.

### 1. Ethical risks
- **Wrongful arrests / legal harms:** Higher false positives for minorities risk unjust detention and conviction.
- **Privacy & chilling effects:** Widespread biometric surveillance undermines civil liberties.
- **Disparate impact:** Amplifies policing in marginalized communities, worsening inequality.
- **Lack of due process & transparency:** Individuals may be misidentified without knowledge or recourse.

### 2. Recommended policies & mitigations
- **Restrict high-stakes deployment.** Do not use as sole probable cause; require corroborative evidence.
- **Human oversight.** Any match must be reviewed by trained personnel and corroborated with other evidence.
- **Independent audits & transparency.** Publish subgroup accuracy metrics and permit independent third-party audits.
- **Data governance & minimization.** Limit retention, require access logs, and protect against misuse.
- **Redress & notification.** Provide mechanisms to notify and correct misidentifications.

---

### Peer Review / Group Notes
- Group members: [Add names here]
- Short reflections: [Each member writes 2–3 lines on their contribution and any open issues.]

"""

def main(outdir='outputs'):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'case_studies.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(CONTENT)
    print(f"Wrote case studies to {path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='outputs')
    args = p.parse_args()
    main(outdir=args.outdir)
# scripts/fairness_audit.py
"""
Run a fairness audit (command-line). Produces outputs in ../outputs/ by default.

Usage:
    python scripts/fairness_audit.py --outdir outputs
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

from utils import demographic_parity, equalized_odds, save_metrics_table

def generate_hiring_data(n=5000, seed=42, female_ratio=0.30):
    np.random.seed(seed)
    gender = np.random.binomial(1, female_ratio, size=n)  # 1 = female
    experience = np.random.normal(5, 2, size=n) + (gender * -0.5)
    education = np.random.choice([0,1,2], p=[0.4,0.4,0.2], size=n)
    skills = np.clip(np.random.normal(0.5 + 0.1*education, 0.15, size=n), 0, 1)
    base_prob = 0.1 + 0.05*experience + 0.15*skills + 0.05*(education==2)
    # Introduce historical bias against females
    prob = 1 / (1 + np.exp(- (base_prob - 0.3*gender)))
    hired = np.random.binomial(1, np.clip(prob, 0, 0.99))
    df = pd.DataFrame({
        'gender': gender,
        'experience': experience,
        'education': education,
        'skills': skills,
        'hired': hired
    })
    return df

def train_and_eval(X_train, y_train, X_test, y_test, A_test, sample_weight=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    return clf, y_prob, y_pred

def find_group_thresholds(probs, y_true, A):
    # choose threshold per group to roughly equalize false negative rates
    thresholds = {}
    groups = np.unique(A)
    for g in groups:
        mask = (A == g)
        t_list = np.linspace(0.0, 1.0, 101)
        fnrs = []
        for t in t_list:
            pred = (probs[mask] >= t).astype(int)
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true[mask], pred, labels=[0,1]).ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fnrs.append((t, fnr))
        thresholds[g] = fnrs
    avg_fnrs = []
    for g in groups:
        arr = np.array([f for (_, f) in thresholds[g]])
        avg_fnrs.append(arr.mean())
    target_fnr = float(np.mean(avg_fnrs))
    chosen = {}
    for g in groups:
        arr = np.array(thresholds[g], dtype=float)
        diffs = np.abs(arr[:,1] - target_fnr)
        idx = int(diffs.argmin())
        chosen[g] = float(thresholds[g][idx][0])
    return chosen

def report_row(name, y_true, y_pred, y_prob, A_test, thresholds=None):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    rates, dp_diff = demographic_parity(y_pred, A_test)
    eo_metrics, eo_diff = equalized_odds(y_pred, y_true.values if hasattr(y_true, 'values') else y_true, A_test)
    row = {
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'auc': auc,
        'sel_rate_female': rates.get(1, None),
        'sel_rate_male': rates.get(0, None),
        'dp_diff_female_minus_male': dp_diff,
        'tpr_diff_female_minus_male': eo_diff.get('tpr_diff'),
        'fpr_diff_female_minus_male': eo_diff.get('fpr_diff'),
        'thresholds': thresholds if thresholds is not None else ''
    }
    return row

def main(outdir='outputs', seed=42):
    os.makedirs(outdir, exist_ok=True)
    df = generate_hiring_data(n=5000, seed=seed)
    X = df[['experience','education','skills']]
    y = df['hired']
    A = df['gender'].values

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.3, random_state=seed, stratify=A
    )

    rows = []

    # Baseline
    clf, y_prob, y_pred = train_and_eval(X_train, y_train, X_test, y_test, A_test)
    rows.append(report_row("Baseline", y_test, y_pred, y_prob, A_test))

    # Mitigation 1: Reweighting (boost female weight)
    sample_weight = np.where(A_train==1, 1.5, 1.0)
    clf_rw, y_prob_rw, y_pred_rw = train_and_eval(X_train, y_train, X_test, y_test, A_test, sample_weight=sample_weight)
    rows.append(report_row("Reweighted", y_test, y_pred_rw, y_prob_rw, A_test))

    # Mitigation 2: Undersampling majority in training
    train_df = X_train.copy()
    train_df['hired'] = y_train.values
    train_df['gender'] = A_train
    major = train_df[train_df['gender']==0]
    minor = train_df[train_df['gender']==1]
    from sklearn.utils import resample
    major_down = resample(major, replace=False, n_samples=len(minor), random_state=seed)
    balanced = pd.concat([major_down, minor])
    X_bal = balanced[['experience','education','skills']]
    y_bal = balanced['hired']
    clf_us = LogisticRegression(max_iter=1000)
    clf_us.fit(X_bal, y_bal)
    y_prob_us = clf_us.predict_proba(X_test)[:,1]
    y_pred_us = (y_prob_us >= 0.5).astype(int)
    rows.append(report_row("Undersample", y_test, y_pred_us, y_prob_us, A_test))

    # Mitigation 3: Post-processing threshold per group (equalize FNR)
    chosen_thresholds = find_group_thresholds(y_prob, y_test.values, A_test)
    y_pred_thresh = np.zeros_like(y_prob).astype(int)
    for i in range(len(y_prob)):
        g = A_test[i]
        t = chosen_thresholds[g]
        y_pred_thresh[i] = 1 if y_prob[i] >= t else 0
    rows.append(report_row("PostThreshold", y_test, y_pred_thresh, y_prob, A_test, thresholds=chosen_thresholds))

    # Save summary
    out_csv = os.path.join(outdir, 'metrics_summary.csv')
    out_json = os.path.join(outdir, 'metrics_summary.json')
    save_metrics_table(rows, out_csv, out_json)
    print(f"Saved metrics summary to {out_csv} and {out_json}")

    # Save DP diff figure
    labels = [r['model'] for r in rows]
    dp_vals = [r['dp_diff_female_minus_male'] for r in rows]
    plt.figure(figsize=(7,4))
    plt.bar(labels, dp_vals)
    plt.ylabel('Demographic parity diff (female - male)')
    plt.title('DP difference across mitigation methods')
    plt.tight_layout()
    fig_path = os.path.join(outdir, 'dp_diff.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved DP figure to {fig_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='outputs', help='Directory to write outputs')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args()
    main(outdir=args.outdir, seed=args.seed)
