import torch
import torch.nn as nn
import torchvision.models as models

CLASSES = ["dog", "cat", "truck", "human", "airplane",
           "duck", "car", "horse", "duck", "ship"]

def model_loader(path: str = "model.pt"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
