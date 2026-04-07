import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.config import OUTPUT_DIR


def plot_results(
    biobert_results: dict,
    baseline_results: dict,
    class_names: list,
) -> None:
    
    hist = biobert_results["history"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Clinical NLP — BioBERT Medical Specialty Classification",
        fontsize=13, fontweight="bold",
    )

    # Training curves
    ax = axes[0]
    epochs = range(1, len(hist["val_acc"]) + 1)
    ax.plot(epochs, hist["val_acc"],   "b-o", label="Val Accuracy", linewidth=2)
    ax.plot(epochs, hist["val_auc"],   "g-o", label="Val AUC-ROC",  linewidth=2)
    ax.plot(epochs, hist["val_f1"],    "m-o", label="Val F1",       linewidth=2)
    ax.plot(epochs, hist["train_loss"],"r--o",label="Train Loss",   linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_title("Training Curves")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(list(epochs))

    # Confusion matrix
    ax = axes[1]
    cm = confusion_matrix(biobert_results["labels"], biobert_results["preds"])
    short = [c.split("/")[0].strip()[:12] for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=short, yticklabels=short)
    ax.set_title("Confusion Matrix — BioBERT")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # Model comparison
    ax = axes[2]
    metrics = ["Accuracy", "F1 Score", "AUC-ROC"]
    b_vals  = [baseline_results["accuracy"], baseline_results["f1"], baseline_results["auc_roc"]]
    bb_vals = [biobert_results["accuracy"],  biobert_results["f1"],  biobert_results["auc_roc"]]
    x = np.arange(3)
    b1 = ax.bar(x - 0.2, b_vals,  0.35, label="TF-IDF + LR", color="#888780")
    b2 = ax.bar(x + 0.2, bb_vals, 0.35, label="BioBERT",     color="#378ADD")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12); ax.set_title("Baseline vs. BioBERT")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "biobert_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Saved: {path}")
    plt.close()


def export_predictions(X_test, biobert_results: dict, class_names: list) -> None:
    
    df = X_test.reset_index(drop=True).to_frame(name="text")
    df["actual"]    = [class_names[l] for l in biobert_results["labels"]]
    df["predicted"] = [class_names[p] for p in biobert_results["preds"]]
    df["confidence"]= biobert_results["proba"].max(axis=1).round(4)
    df["correct"]   = (biobert_results["labels"] == biobert_results["preds"]).astype(int)
    for i, cls in enumerate(class_names):
        safe = cls.split("/")[0].strip().replace(" ","_")[:12]
        df[f"prob_{safe}"] = biobert_results["proba"][:, i].round(4)

    path = os.path.join(OUTPUT_DIR, "predictions_tableau.csv")
    df.to_csv(path, index=False)
    print(f"[Export] Saved: {path}")