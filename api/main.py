import os
from types import prepare_class

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
import torch, io
from api.model_loader import model_loader, CLASSES
from api.logger import log_prediction
import uvicorn
app = FastAPI(title="again Project 2")
model = model_loader("model.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class PredictResponse(BaseModel):
    predicted_class: int
    class_index: int
    confidence: float

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.argmax(outputs, dim=1)
        idx = probs.argmax(dim=1).item()
        conf = round(probs[0][idx].item(), 4)
    log_prediction(len(contents), CLASSES[idx], conf)

    return {
        "predicted_class": CLASSES[idx],
        "class_index": idx,
        "confidence": conf
    }

if __name__ == 'main':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host = "0.0.0.0", port=port)
