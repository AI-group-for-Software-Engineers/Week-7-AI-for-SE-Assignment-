# scripts/utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def demographic_parity(y_hat, A):
    """Return selection rates per protected group and difference (female - male)."""
    groups = np.unique(A)
    rates = {int(g): float(np.mean(y_hat[A == g])) for g in groups}
    diff = rates.get(1, 0.0) - rates.get(0, 0.0)
    return rates, diff

def disparate_impact(y_hat, A):
    rates, _ = demographic_parity(y_hat, A)
    denom = rates.get(0, 0)
    if denom == 0:
        return rates, float('nan')
    return rates, rates.get(1, 0) / denom

def equalized_odds(y_hat, y_true, A):
    groups = np.unique(A)
    metrics = {}
    for g in groups:
        mask = (A == g)
        if mask.sum() == 0:
            metrics[int(g)] = {'tpr': np.nan, 'fpr': np.nan}
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_hat[mask], labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        metrics[int(g)] = {'tpr': float(tpr), 'fpr': float(fpr)}
    try:
        tpr_diff = metrics[1]['tpr'] - metrics[0]['tpr']
        fpr_diff = metrics[1]['fpr'] - metrics[0]['fpr']
    except Exception:
        tpr_diff = fpr_diff = np.nan
    return metrics, {'tpr_diff': tpr_diff, 'fpr_diff': fpr_diff}

def save_metrics_table(rows, out_csv_path, out_json_path):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    df.to_json(out_json_path, orient='records', indent=2)
    return df
