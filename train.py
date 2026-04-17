import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import mlflow.pytorch
import json

from torchvision.models import ResNet18_Weights

LR = 0.001
EPOCHS = 3
BATCH_SIZE = 32
DEVICE = torch.device("cpu" if torch.backends.mps.is_available()
                      else "gpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.CIFAR10(root="./data", train=True,
                           download=True, transform=transform)
small_set = Subset(dataset, range(5000))
loader = DataLoader(small_set, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)

optimizer = optim.Adam(model.fc.parameters(), lr=LR)
criterion = CrossEntropyLoss()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("final-project-resnet")

with mlflow.start_run():
    mlflow.log_params({
        "lr": LR, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "model": "resnet18"
    })
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0
        for X,y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss      += loss.item()
            total_correct   += (outputs.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss/len(small_set)
        accuracy = total_correct/len(loader)

        mlflow.log_metrics({
            "loss": avg_loss,
            "accuracy": accuracy
        }, step=epoch)
        torch.save(model.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")
        metrics = {"final_loss": avg_loss, "accuracy": accuracy}
        with open ("metrics.json", "w") as f:
            json.dump(metrics, f)

    print(f'Обучение завершено. Команда ддя проверки:'
          f'mlflow ui --backend-store-uri sqlite:///mlflow.db')