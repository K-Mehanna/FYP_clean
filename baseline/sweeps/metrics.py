"""Shared classification metrics for the baseline sweeps.

Lifted from ``baseline/end-to-end/baseline_resnet_brainwear_cv.ipynb`` (cell
``d82c1a94``) and generalised so the label space is explicit instead of the
hardcoded ``[0, 1, 2, 3, 4]``. Both the end-to-end and autoencoder-classifier
trainers import ``print_metrics`` from here so the two pipelines report metrics
identically (in particular, balanced accuracy).
"""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def print_metrics(
    y_true,
    y_pred,
    strategy: str,
    header: str,
    num_labels: int,
    include_balanced: bool = False,
) -> dict:
    """Print and return a classification metrics bundle.

    Args:
        y_true, y_pred: array-likes of integer class labels.
        strategy: "categorical" or "ordinal" (recorded for context).
        header: title printed above the report.
        num_labels: size of the label space (``range(num_labels)``). For the
            end-to-end models this is the number of model outputs (5 for
            categorical, since the head always emits 5 logits); for the
            autoencoder classifier it is ``num_classes``.
        include_balanced: also compute balanced accuracy + normalised CM.
    """
    labels = list(range(num_labels))
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = (
        confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        if include_balanced
        else None
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if include_balanced else None

    print(f"--- {header} ({strategy.upper()}) ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    if include_balanced:
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        print("Normalized confusion matrix (rows=true, cols=pred):")
        print(cm_normalized)
    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3))

    macro = report.get("macro avg", {})
    return {
        "strategy": strategy,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(macro.get("precision", 0.0)),
        "recall": float(macro.get("recall", 0.0)),
        "f1": float(macro.get("f1-score", 0.0)),
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": None
        if cm_normalized is None
        else cm_normalized.tolist(),
        "balanced_accuracy": None if balanced_acc is None else float(balanced_acc),
        "classification_report": report,
    }
