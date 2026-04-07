import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    class_names: list,
) -> dict:
    
    print("[Baseline] Training TF-IDF + Logistic Regression...")

    tfidf = TfidfVectorizer(
        max_features=25_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    X_tr_vec = tfidf.fit_transform(X_train)
    X_te_vec = tfidf.transform(X_test)

    lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
    lr.fit(X_tr_vec, y_train)

    preds = lr.predict(X_te_vec)
    proba = lr.predict_proba(X_te_vec)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=class_names))

    return {
        "model":    lr,
        "tfidf":    tfidf,
        "accuracy": acc,
        "f1":       f1,
        "auc_roc":  auc,
        "preds":    preds,
        "proba":    proba,
    }