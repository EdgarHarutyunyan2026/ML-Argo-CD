FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mlflow \
    boto3 \
    pydantic

COPY app-version.py .
COPY model.py .

EXPOSE 8000

CMD ["uvicorn", "app-version:app", "--host", "0.0.0.0", "--port", "8000"]
