"""
Evaluation metrics and statistical analysis for violence behavior recognition.
Calculates Precision, Recall, F1, AUROC, AUPRC, False Alarm Rate, and optimal thresholds.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auroc: float
    auprc: float
    false_alarm_rate: float  # False Positive Rate (FP / (FP + TN))
    miss_rate: float  # False Negative Rate (FN / (TP + FN))
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    threshold: float
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "===========================================================",
            f"  Behavior Classification Performance (Threshold = {self.threshold:.3f})",
            "===========================================================",
            f"  Total Samples     : {self.total_samples}",
            f"  Accuracy          : {self.accuracy * 100:.2f}%",
            f"  Precision         : {self.precision * 100:.2f}%",
            f"  Recall (Sens.)    : {self.recall * 100:.2f}%",
            f"  Specificity       : {self.specificity * 100:.2f}%",
            f"  F1-Score          : {self.f1:.4f}",
            f"  AUROC             : {self.auroc:.4f}",
            f"  AUPRC             : {self.auprc:.4f}",
            f"  False Alarm Rate  : {self.false_alarm_rate * 100:.2f}% (FPR)",
            f"  Miss Rate         : {self.miss_rate * 100:.2f}% (FNR)",
            "-----------------------------------------------------------",
            "  Confusion Matrix  :",
            f"    TP: {self.true_positives:<6} | FP: {self.false_positives:<6}",
            f"    FN: {self.false_negatives:<6} | TN: {self.true_negatives:<6}",
            "===========================================================",
        ]
        return "\n".join(lines)


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> ClassificationMetrics:
    """
    Computes full suite of classification and operational monitoring metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Safe AUROC / AUPRC calculation when single class is present
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_probs))
        auprc = float(average_precision_score(y_true, y_probs))
    else:
        auroc = 0.5
        auprc = float(np.mean(y_true))

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        specificity=float(spec),
        f1=float(f1),
        auroc=float(auroc),
        auprc=float(auprc),
        false_alarm_rate=float(far),
        miss_rate=float(miss),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        threshold=float(threshold),
        total_samples=len(y_true),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    criterion: str = "max_f1",
    max_acceptable_fpr: float = 0.05,
) -> Tuple[float, ClassificationMetrics]:
    """
    Finds the optimal decision threshold based on specified operational criteria:
    - 'max_f1': Maximizes the balance between precision and recall.
    - 'constrained_fpr': Maximizes recall while guaranteeing false positive rate <= max_acceptable_fpr.
    - 'balanced_accuracy': Maximizes (Sensitivity + Specificity) / 2 (Youden's J statistic).
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    thresholds = np.linspace(0.01, 0.99, 199)
    best_thresh = 0.50
    best_score = -1.0
    best_metrics = evaluate_predictions(y_true, y_probs, threshold=0.50)

    for th in thresholds:
        m = evaluate_predictions(y_true, y_probs, threshold=th)
        if criterion == "max_f1":
            score = m.f1
        elif criterion == "balanced_accuracy":
            score = (m.recall + m.specificity) / 2.0
        elif criterion == "constrained_fpr":
            if m.false_alarm_rate <= max_acceptable_fpr:
                score = m.recall
            else:
                score = -1.0
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_thresh = th
            best_metrics = m

    return best_thresh, best_metrics


def calculate_detection_delay(
    video_events: List[Dict[str, Any]],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Measures clip-level detection delay across multiple annotated test sequences.
    Each event in video_events contains:
      - 'clip_timestamps': list of floats (seconds)
      - 'predicted_probs': list of floats
      - 'ground_truth_start': float (onset of violence in seconds)
      - 'ground_truth_end': float (end of violence in seconds)
    """
    delays = []
    missed_events = 0
    total_events = len(video_events)

    for event in video_events:
        ts = np.array(event["clip_timestamps"])
        probs = np.array(event["predicted_probs"])
        gt_start = event["ground_truth_start"]

        # Clip windows that occur during the event
        incident_mask = probs >= threshold
        incident_indices = np.where(incident_mask)[0]

        if len(incident_indices) > 0:
            first_detection_time = ts[incident_indices[0]]
            delay = max(0.0, first_detection_time - gt_start)
            delays.append(delay)
        else:
            missed_events += 1

    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    p95_delay = float(np.percentile(delays, 95)) if delays else float("nan")

    return {
        "total_events": total_events,
        "detected_events": len(delays),
        "missed_events": missed_events,
        "mean_delay_seconds": mean_delay,
        "median_delay_seconds": median_delay,
        "p95_delay_seconds": p95_delay,
    }
