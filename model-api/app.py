import os
import torch
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import MyNeuralNet


MODEL_NAME="model-v3"

# Внутри minikube используем cluster-internal адреса
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio.mlops.svc.cluster.local:9000"

#mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
mlflow.set_tracking_uri("http://mlflow.local")

device = torch.device("cpu")  # minikube без GPU

def load_model_from_mlflow():
    print("Подключаемся к MLflow...")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("mnist_classification")
    if experiment is None:
        raise Exception("Эксперимент не найден")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise Exception("Нет запусков в эксперименте")

    run_id = runs[0].info.run_id
    print(f"Найден run: {run_id}")

    #model_uri = f"runs:/{run_id}/model-v7"
    model_uri = f"runs:/{run_id}/{MODEL_NAME}"
    print(f"Загружаем модель из: {model_uri}")

    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()
    print("Модель загружена!")
    return model

model = load_model_from_mlflow()

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

        tensor = torch.tensor(data.pixels, dtype=torch.float32)
        tensor = tensor.view(1, 1, 28, 28).to(device)

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

