import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_report():
    reference = pd.DataFrame({
        "image_size": np.random.normal(50000, 10000, 500),
        "confidence": np.random.normal(0.6, 0.15, 500)
    })
    current = pd.read_csv("logs/predictions.csv")[[
        "image_size", "confidence"
    ]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("monitoring/drift_report.html")
    print(f'Report сохранен. Го в monitoring/')

if __name__ == "__main__":
    generate_report()