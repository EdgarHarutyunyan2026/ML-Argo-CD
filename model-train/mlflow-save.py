# mlflow-save.py
import os
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------
# Настройка окружения MLflow / MinIO
# -----------------------
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-api.local"

mlflow.set_tracking_uri("http://mlflow.local")

# -----------------------
# Инициализация клиента
# -----------------------
client = MlflowClient()

# -----------------------
# Имя модели и эксперимента
# -----------------------
model_name = "mnist_classification"
experiment_name = "mnist_classification"

# -----------------------
# Создаём Registered Model, если её нет
# -----------------------
try:
    client.get_registered_model(model_name)
    print(f"Registered Model '{model_name}' уже существует")
except mlflow.exceptions.RestException:
    client.create_registered_model(model_name)
    print(f"Registered Model '{model_name}' создана")

# -----------------------
# Берём последний run
# -----------------------
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise RuntimeError(f"Эксперимент '{experiment_name}' не найден")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError(f"Нет запусков в эксперименте '{experiment_name}'")

latest_run = runs[0]
run_id = latest_run.info.run_id
print(f"Используем последний run: {run_id}")

# -----------------------
# Создаём версию модели
# -----------------------
model_version = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model-v4",
    run_id=run_id
)
print(f"Создана версия модели: {model_version.version}")

# -----------------------
# Переводим версию в Production
# -----------------------
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
print(f"Модель {model_name} версии {model_version.version} переведена в Production")
