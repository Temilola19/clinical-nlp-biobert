import torch
from src.data.loader     import load_and_clean, add_keyword_features, encode_and_split
from src.models.baseline import train_and_evaluate as train_baseline
from src.models.biobert  import BioBERTTrainer
from src.viz.charts      import plot_results, export_predictions
from src.utils.config    import MODEL_NAME


def main():
    print("=" * 60)
    print("CLINICAL NLP — BIOBERT SPECIALTY CLASSIFICATION")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Load and prepare data
    df = load_and_clean()
    df = add_keyword_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test, le = encode_and_split(df)
    class_names = list(le.classes_)

    #  Baseline
    baseline = train_baseline(X_train, X_test, y_train, y_test, class_names)

    # BioBERT fine-tuning
    trainer = BioBERTTrainer(num_classes=len(class_names))
    trainer.train(X_train, y_train, X_val, y_val)

    #  Evaluate
    biobert = trainer.evaluate(X_test, y_test, class_names)

    #  Print improvement
    print(f"\n  Accuracy improvement: "
          f"+{(biobert['accuracy'] - baseline['accuracy'])*100:.1f}pp")
    print(f"  AUC-ROC improvement:  "
          f"+{(biobert['auc_roc']  - baseline['auc_roc']):.3f}")

    #  Save model
    trainer.save()

    #  Visualizations + export
    plot_results(biobert, baseline, class_names)
    export_predictions(X_test, biobert, class_names)

    #  Demo inference
    sample = (
        "Patient with chest pain and shortness of breath. "
        "EKG shows ST elevation. Troponin elevated. Heparin drip started."
    )
    result = trainer.predict_text(sample, class_names)
    print(f"\n  Sample inference:")
    print(f"    Note:       '{sample[:60]}...'")
    print(f"    Predicted:  {result['predicted']} ({result['confidence']:.1%})")

    print("\n  Done — outputs saved to ./outputs/")


if __name__ == "__main__":
    main()