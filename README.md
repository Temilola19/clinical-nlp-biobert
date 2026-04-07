# Clinical NLP — Medical Specialty Classification

A comparative study fine-tuning BioBERT on real clinical transcription notes to classify medical specialties from unstructured text. The core question this project answers is simple: does domain-specific pretraining actually matter for clinical text, and by how much?

The short answer is yes, meaningfully — BioBERT outperforms a well-tuned TF-IDF baseline by around 14 percentage points on this task. The longer answer is in the results below.

---

## What it does

The pipeline takes raw medical transcription notes from the MTSamples dataset, trains a TF-IDF + Logistic Regression baseline, fine-tunes BioBERT on the same data, and produces a side-by-side comparison with full evaluation metrics. The final model can classify a new clinical note into one of six specialty categories in a single forward pass.

**The six categories:**
Cardiovascular / Pulmonary · Neurology · Gastroenterology · Orthopedic · Radiology · Discharge Summary

---

## Project structure

```
project2_clinical_nlp/
├── main.py                    # entry point — run this
├── requirements.txt
├── src/
│   ├── data/
│   │   └── loader.py          # load, clean, feature engineering, split
│   ├── models/
│   │   ├── baseline.py        # TF-IDF + Logistic Regression
│   │   └── biobert.py         # BioBERT Dataset, Trainer, evaluation, inference
│   ├── viz/
│   │   └── charts.py          # training curves, confusion matrix, comparison bar chart
│   └── utils/
│       └── config.py          # all configuration in one place
└── outputs/                   # generated after running
```

---

## Getting started

**Step 1 — Get the data**

Download MTSamples from Kaggle (free account required):
[kaggle.com/datasets/tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

Place `mtsamples.csv` in the `data/` folder.

**Step 2 — Install and run**

```bash
git clone https://github.com/Temilola19/clinical-nlp
cd clinical-nlp
pip install -r requirements.txt
python main.py
```

Training takes around 15–20 minutes on CPU, or 3–5 minutes with a GPU. Google Colab with a free T4 GPU works well for this.

---

## Results

| Model | Accuracy | F1 (weighted) | AUC-ROC |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~73% | ~0.72 | ~0.91 |
| BioBERT (fine-tuned) | ~87% | ~0.86 | ~0.96 |

Results vary slightly by run. The improvement comes from BioBERT's pretraining on 18 billion words of biomedical text — it understands that "troponin" and "ST elevation" belong together in a way that word-frequency methods cannot capture.

---

## Outputs

| File | Description |
|---|---|
| `biobert_results.png` | Training curves, confusion matrix, baseline vs BioBERT comparison |
| `predictions_tableau.csv` | Per-note predictions with confidence scores for each class |
| `saved_model/` | Fine-tuned model weights — load with HuggingFace `from_pretrained()` |

---

## Inference

Once trained, classifying new notes is straightforward:

```python
from src.models.biobert import BioBERTTrainer

trainer = BioBERTTrainer(num_classes=6)
# load saved weights
result = trainer.predict_text(
    "Patient presents with chest pain. EKG shows ST elevation. Troponin elevated.",
    class_names=["Cardiovascular / Pulmonary", "Neurology", ...]
)
print(result["predicted"])   # → "Cardiovascular / Pulmonary"
print(result["confidence"])  # → 0.94
```

---

## Tech stack

Python · PyTorch · HuggingFace Transformers · BioBERT · scikit-learn · pandas · Matplotlib · Seaborn
