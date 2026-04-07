import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import (
    MTSAMPLES_PATH, TARGET_CATEGORIES, SEED, TEST_SIZE, VAL_SIZE
)

CLINICAL_KEYWORDS = {
    "cardiac_terms":    ["heart", "cardiac", "coronary", "artery", "echocardiogram"],
    "neuro_terms":      ["brain", "seizure", "stroke", "neurolog", "headache"],
    "gi_terms":         ["abdomen", "gastro", "bowel", "colonoscopy", "liver"],
    "ortho_terms":      ["fracture", "joint", "spine", "orthopedic", "knee"],
    "radiology_terms":  ["x-ray", "imaging", "radiolog", "scan", "contrast"],
    "discharge_terms":  ["discharge", "hospital course", "admitted", "diagnosis"],
}


def load_and_clean() -> pd.DataFrame:
  
    if not pd.io.common.file_exists("raw/mtsamples.csv"):
        raise FileNotFoundError(
         "MTSamples CSV not found at raw/mtsamples.csv.\n"
        "Download: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
         "Place 'mtsamples.csv' in the data/ folder."
    )

    df = pd.read_csv("raw/mtsamples.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[["medical_specialty", "transcription"]].copy()
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip()
    df["text"]  = df["text"].str.strip().str[:1000]
    df = df[df["text"].str.len() > 100]
    df = df[df["label"].isin(TARGET_CATEGORIES)].reset_index(drop=True)

    print(f"[Data] Loaded {len(df)} notes | {df['label'].nunique()} specialties")
    print(df["label"].value_counts().to_string())
    return df


def add_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary clinical keyword presence features."""
    for feat, kws in CLINICAL_KEYWORDS.items():
        df[feat] = df["text"].str.lower().apply(
            lambda x: int(any(kw in x for kw in kws))
        )
    return df


def encode_and_split(df: pd.DataFrame):
   
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        df["text"], df["label_id"],
        test_size=TEST_SIZE, stratify=df["label_id"], random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=VAL_SIZE, stratify=y_tmp, random_state=SEED
    )
    print(f"[Data] Split — Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te, le