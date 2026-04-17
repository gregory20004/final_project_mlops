import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/prediction.csv"

def log_prediction(image_size: int, prediction_class: int, confidence: float):
    os.makedirs("logs", exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(),
        "image_size": image_size,
        "prediction_class": prediction_class,
        "confidence": confidence
    }
    df = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=True, index=False)