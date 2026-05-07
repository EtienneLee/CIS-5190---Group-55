# CIS 5190 Group 55 — News Headline Classifier

Binary classification of Fox News vs NBC News headlines using fine-tuned RoBERTa + LinearSVC.

## Files
- `model.py` — NewsClassifier class with LinearSVC classification head
- `preprocess.py` — prepare_data function for inference
- `model_training_pipeline.ipynb` — full transformer fine-tuning pipeline (RoBERTa, DeBERTa)
- `svm.py` — trains LinearSVC on top of fine-tuned transformer embeddings
- `bake_svm.py` — bakes SVM coefficients into model weights for single-file submission

## Model Weights
Both weight files are available on HuggingFace:
- `model.pt` — fine-tuned RoBERTa-base backbone weights
- `model_final.pt` — submission file, same backbone with SVM coefficients baked in

## Links
- HuggingFace Dataset + Weights: https://huggingface.co/datasets/PudgySquirrel/cis5190-news-headlines
