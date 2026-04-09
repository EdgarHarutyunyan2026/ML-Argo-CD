# 🚀 MLOps Demo Project (MNIST Classification)

## 📌 Описание проекта

Это демонстрационный **MLOps-пайплайн**, который показывает полный жизненный цикл ML-модели:

* Обучение модели (MNIST классификация)
* Логирование экспериментов в MLflow
* Хранение артефактов в MinIO (S3-compatible)
* Деплой модели как API-сервиса
* Автоматическое обновление через ArgoCD (GitOps)

Проект приближен к реальному production-подходу и демонстрирует, как строится современная ML-инфраструктура.

---

## 🏗️ Архитектура

```text
Model Training → MLflow → MinIO
                      ↓
                  Model API (FastAPI + Uvicorn)
                      ↓
                Kubernetes (Helm)
                      ↓
                  ArgoCD (GitOps)
```

---

## 📂 Структура проекта

```bash
.
├── argo-cd/          # Конфигурация ArgoCD (GitOps)
├── helm/             # Helm chart для деплоя всех сервисов
├── model-api/        # API для инференса модели
└── model-train/      # Обучение модели и логирование
```

### 🔹 `model-train`

* `train.py` — обучение модели MNIST
* Параметры (learning rate, batch size, epochs) задаются через переменные
* Сохраняет:

  * метрики в **MLflow**
  * артефакты модели в **MinIO**

---

### 🔹 `model-api`

* `app.py` — API сервис для инференса
* Загружает модель из **MLflow**
* Запускается через **Uvicorn**
* Принимает входные данные (массив пикселей) и возвращает предсказание

---

### 🔹 `helm`

* Helm chart для деплоя:

  * MLflow
  * MinIO
  * Model API
  * Ingress
* values-файлы:

  * `mlflow-values.yaml`
  * `minio-values.yaml`
  * `mnist-app-values.yaml`

---

### 🔹 `argo-cd`

* Конфигурация для **GitOps**
* Следит за репозиторием
* Автоматически обновляет кластер при изменениях

---

## ⚙️ Как работает пайплайн

### 1. 📦 Поднимаем инфраструктуру

```bash
# MinIO (хранилище артефактов)
helm install minio ...

# Создаём bucket для моделей

# MLflow
helm install mlflow ...
```

---

### 2. 🧠 Обучение модели

```bash
cd model-train
python train.py
```

Во время обучения:

* логируются эксперименты в MLflow
* модель сохраняется в MinIO

---

### 3. 🧪 Тестирование

```bash
python test.py
```

Проверяем качество модели перед деплоем.

---

### 4. 🐳 Сборка API

```bash
cd model-api
docker build -t mnist-api:latest .
```

* В `app.py` указывается имя модели из MLflow
* API при запуске автоматически загружает модель

---

### 5. ☸️ Деплой в Kubernetes

```bash
helm install mnist-app ./helm/argo-ml
```

---

### 6. 🔁 GitOps (ArgoCD)

* ArgoCD следит за репозиторием
* При изменении:

  * версии образа
  * конфигурации Helm

👉 автоматически делает redeploy

---

## 🔄 Обновление модели (Production flow)

1. Обучаем новую модель
2. Проверяем качество
3. Обновляем версию модели в `app.py`
4. Собираем новый Docker image
5. Обновляем тег в Helm values
6. Пушим изменения в Git

✅ ArgoCD автоматически обновляет приложение

---

## 🧰 Используемые технологии

* Python
* PyTorch / ML stack
* MLflow
* MinIO (S3 storage)
* FastAPI + Uvicorn
* Docker
* Kubernetes
* Helm
* ArgoCD (GitOps)

---

## 🎯 Цель проекта

Показать:

* как строится end-to-end ML pipeline
* как внедряется GitOps в ML
* как автоматизировать деплой моделей
* как приблизить ML-проект к production

---

## 📌 Примечание

Это demo-проект, но архитектура максимально приближена к реальным MLOps системам.

---

## 👨‍💻 Автор

Demo MLOps project для практики и демонстрации навыков работы с:

* Kubernetes
* ML lifecycle
* CI/CD и GitOps

