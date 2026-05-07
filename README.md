# CIS 5190 Group 55 — News Headline Classifier

Binary classification of Fox News vs NBC News headlines using fine-tuned RoBERTa + LinearSVC.

## Core Submission Files
The three files required for evaluation are `model.py`, `preprocess.py`, and `model_final.pt` (available on HuggingFace below).

## Repository Files
- `model.py` — NewsClassifier class with LinearSVC classification head
- `preprocess.py` — prepare_data function for inference
- `model_training_pipeline.ipynb` — full transformer fine-tuning pipeline (RoBERTa, DeBERTa)
- `svm.py` — trains LinearSVC on top of fine-tuned transformer embeddings
- `bake_svm.py` — bakes SVM coefficients into model weights for single-file submission

Note: scraping scripts are not included here; data collection methodology is described in the project report.

## HuggingFace
Model weights and training dataset are hosted on HuggingFace as they exceeded GitHub's file size limits:
https://huggingface.co/datasets/PudgySquirrel/cis5190-news-headlines

- `model.pt` — fine-tuned RoBERTa-base backbone only (no SVM)
- `model_final.pt` — submission file, backbone + SVM coefficients baked in. Load this for inference.
- `classified_headline.csv` — full training dataset (Fox + NBC, 17,897 headlines)
