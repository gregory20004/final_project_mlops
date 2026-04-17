from fastapi.testclient import TestClient
from api.main import app
from PIL import Image
import io

client = TestClient(app)

def test_heath():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    img = Image.new("RGB", (224, 224), color=(100,150,200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file":("anything.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1