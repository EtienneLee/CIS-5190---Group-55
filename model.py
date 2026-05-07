from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
import pickle

MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 64

class NewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(self.device_type)
        
        self.register_buffer("svm_coef", torch.zeros(1, 768))
        self.register_buffer("svm_intercept", torch.zeros(1))
        self.svm_loaded = False
        
        if os.path.exists("svm.pkl"):
            with open("svm.pkl", "rb") as f:
                svm = pickle.load(f)
            self.svm_coef = nn.Parameter(
                torch.tensor(svm.coef_, dtype=torch.float32), requires_grad=False)
            self.svm_intercept = nn.Parameter(
                torch.tensor(svm.intercept_, dtype=torch.float32), requires_grad=False)
            self.svm_loaded = True
            print("SVM loaded from pkl")
        
        self.eval()

    def get_cls_embeddings(self, batch):
        enc = self.tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(self.device_type) for k, v in enc.items()}
        with torch.no_grad():
            output = self.backbone.roberta(**enc)
            cls = output.last_hidden_state[:, 0, :].cpu()
        return cls

    @torch.no_grad()
    def predict(self, batch):
        if isinstance(batch, str):
            batch = [batch]
        batch = [str(x) if x is not None else "" for x in batch]

        if self.svm_coef.abs().sum() > 0:
            all_preds = []
            chunk_size = 64
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i+chunk_size]
                emb = self.get_cls_embeddings(chunk)
                scores = emb @ self.svm_coef.T + self.svm_intercept
                preds = (scores.squeeze() > 0).long().cpu().tolist()
                if isinstance(preds, int):
                    preds = [preds]
                all_preds.extend(preds)
            return all_preds
        else:
            all_preds = []
            chunk_size = 64
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i+chunk_size]
                enc = self.tokenizer(chunk, padding=True, truncation=True,
                                    max_length=MAX_LENGTH, return_tensors="pt")
                enc = {k: v.to(self.device_type) for k, v in enc.items()}
                logits = self.backbone(**enc).logits
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
            return all_preds

def get_model():
    return NewsClassifier()

Model = NewsClassifier