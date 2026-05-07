import torch
from model import NewsClassifier

model = NewsClassifier()  # loads svm.pkl automatically
state_dict = torch.load("model.pt", map_location="cpu")

# load roberta weights
model.load_state_dict(state_dict, strict=False)

# save everything including svm buffers
torch.save(model.state_dict(), "model_final.pt")
print("Saved model_final.pt with SVM baked in")