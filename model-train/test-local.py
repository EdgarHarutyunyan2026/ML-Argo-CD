import mlflow
import mlflow.pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import os

# -----------------------
# Настройка
# -----------------------
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-api.local"

mlflow.set_tracking_uri("http://mlflow.local")

EXPERIMENT_NAME = "mnist_classification"
MODEL_NAME = "model-v1" 

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Находим run с нужной моделью
# -----------------------
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=20
)

run_id = None
for run in runs:
    inner = client.list_artifacts(run.info.run_id, path=MODEL_NAME)
    if inner:
        run_id = run.info.run_id
        print(f"Найдено в run: {run.info.run_name}")
        break

if not run_id:
    print(f"❌ Модель '{MODEL_NAME}' не найдена ни в одном run!")
    exit(1)

# -----------------------
# Загрузка и тест
# -----------------------
model_uri = f"runs:/{run_id}/{MODEL_NAME}"
print(f"Загружаем: {model_uri}")

model = mlflow.pytorch.load_model(model_uri)
model = model.to(device)
model.eval()
print("Модель загружена!")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
image, label = dataset[71]

with torch.no_grad():
    input_tensor = image.unsqueeze(0).to(device)
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

print(f"\nРеальная цифра:       {label}")
print(f"Предсказание модели:  {predicted_class.item()}")
print(f"Уверенность:          {confidence.item():.2%}")
