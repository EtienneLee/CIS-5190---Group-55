import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import time

MODEL_NAME = "FacebookAI/roberta-base"
WEIGHTS_PATH = "model.pt"
MAX_LENGTH = 64
DATA_PATH = "classified_headline.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load backbone with fine-tuned weights
backbone = AutoModel.from_pretrained(MODEL_NAME)
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
roberta_weights = {k.replace("backbone.roberta.", ""): v 
                   for k, v in state_dict.items() 
                   if "backbone.roberta." in k}
backbone.load_state_dict(roberta_weights, strict=False)
backbone.eval()
backbone = backbone.to(device)
print("Backbone loaded")

# load data
df = pd.read_csv(DATA_PATH)
headlines = df['headline'].fillna("").astype(str).tolist()
labels = df['label'].tolist()
print(f"Loaded {len(headlines)} headlines")

def get_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            output = backbone(**enc)
            cls = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls)
        if i % 500 == 0:
            print(f"  Embeddings: {i}/{len(texts)}")
    return np.vstack(all_embeddings)

print("Extracting embeddings...")
start = time.time()
embeddings = get_embeddings(headlines)
print(f"Done in {time.time()-start:.1f}s, shape: {embeddings.shape}")

np.save("embeddings.npy", embeddings)
np.save("labels.npy", np.array(labels))
print("Embeddings saved")

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

print("Training SVM...")
svm = LinearSVC(max_iter=2000)
svm.fit(X_train, y_train)

preds = svm.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")

with open("svm.pkl", "wb") as f:
    pickle.dump(svm, f)
print("SVM saved to svm.pkl")