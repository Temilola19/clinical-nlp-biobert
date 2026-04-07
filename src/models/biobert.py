import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm

from src.utils.config import (
    MODEL_NAME, MAX_LEN, BATCH_SIZE, EPOCHS, LR, SEED, MODEL_DIR
)


class ClinicalDataset(Dataset):
    """PyTorch Dataset wrapping tokenized clinical text."""

    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts   = list(texts)
        self.labels  = list(labels)
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BioBERTTrainer:
    """Encapsulates BioBERT fine-tuning and evaluation."""

    def __init__(self, num_classes: int, device: torch.device = None):
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model       = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_classes
        ).to(self.device)
        self.history     = {"train_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}
        print(f"[BioBERT] Model loaded: {MODEL_NAME} | Device: {self.device}")

    def _make_loader(self, texts, labels, shuffle: bool) -> DataLoader:
        ds = ClinicalDataset(texts, labels, self.tokenizer)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    def train(self, X_train, y_train, X_val, y_val):
        """Fine-tune BioBERT with linear warmup scheduling."""
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

        optimizer   = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        print(f"[BioBERT] Training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False):
                optimizer.zero_grad()
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"].to(self.device)
                loss = self.model(ids, attention_mask=mask, labels=labs).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss        = total_loss / len(train_loader)
            vl, vp, vproba  = self._predict(val_loader)
            val_acc = accuracy_score(vl, vp)
            val_f1  = f1_score(vl, vp, average="weighted")
            val_auc = roc_auc_score(vl, vproba, multi_class="ovr", average="weighted")

            self.history["train_loss"].append(avg_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["val_auc"].append(val_auc)
            print(f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

    def evaluate(self, X_test, y_test, class_names: list) -> dict:
        """Full test-set evaluation with classification report."""
        loader = self._make_loader(X_test, y_test, shuffle=False)
        labels, preds, proba = self._predict(loader)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted")
        auc = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")

        print(f"\n[BioBERT] Test Results:")
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
        print(classification_report(labels, preds, target_names=class_names))

        return {"accuracy": acc, "f1": f1, "auc_roc": auc,
                "preds": preds, "labels": labels, "proba": proba,
                "history": self.history}

    def predict_text(self, text: str, class_names: list) -> dict:
        """Single-note inference — usable in a Flask/Streamlit API."""
        self.model.eval()
        enc = self.tokenizer(
            text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            ).logits
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {
            "predicted":   class_names[np.argmax(proba)],
            "confidence":  float(proba.max()),
            "all_probs":   dict(zip(class_names, proba.round(3).tolist())),
        }

    def save(self):
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        print(f"[BioBERT] Model saved → {MODEL_DIR}")

    def _predict(self, loader):
        self.model.eval()
        all_labels, all_preds, all_proba = [], [], []
        with torch.no_grad():
            for batch in loader:
                ids   = batch["input_ids"].to(self.device)
                mask  = batch["attention_mask"].to(self.device)
                labs  = batch["labels"]
                proba = torch.softmax(
                    self.model(ids, attention_mask=mask).logits, dim=1
                ).cpu().numpy()
                all_labels.extend(labs.numpy())
                all_preds.extend(np.argmax(proba, axis=1))
                all_proba.extend(proba)
        return np.array(all_labels), np.array(all_preds), np.array(all_proba)