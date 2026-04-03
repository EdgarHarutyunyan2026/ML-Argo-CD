import os
import torch
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# -------------------------
# Настройки окружения для minikube / MLflow
# -------------------------
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio.mlops.svc.cluster.local:9000"

# MLflow tracking server (cluster-internal DNS)
mlflow.set_tracking_uri("http://mlflow.local")

# Устройство для PyTorch
device = torch.device("cpu")  # minikube без GPU

# -------------------------
# Функция загрузки последней модели из MLflow
# -------------------------
def load_model_from_mlflow():
    print("Подключаемся к MLflow...")

    # Загружаем модель из stage "Production" (или "Latest", если нет stage)
    model_uri = "models:/mnist_classification/Production"
    print(f"Загружаем модель из MLflow: {model_uri}")

    try:
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        print("Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise RuntimeError(f"Не удалось загрузить модель из MLflow: {e}")

# Загружаем модель один раз при старте API
model = load_model_from_mlflow()

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="MNIST Classifier")

class ImageInput(BaseModel):
    pixels: List[float]

class PredictionOutput(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: List[float]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: ImageInput):
    try:
        if len(data.pixels) != 784:
            raise HTTPException(status_code=400, detail=f"Ожидается 784 пикселя, получено {len(data.pixels)}")

        tensor = torch.tensor(data.pixels, dtype=torch.float32).view(1, 1, 28, 28).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return PredictionOutput(
            predicted_class=predicted.item(),
            confidence=round(confidence.item(), 4),
            probabilities=probabilities.squeeze().tolist()
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
