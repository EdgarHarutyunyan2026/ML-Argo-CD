# test.py
import requests
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Укажи полный путь где уже скачан MNIST
dataset = torchvision.datasets.MNIST(
    #root="/home/edgar/DevOps/MLOps/k8s/pytorch/model/data", 
    root="/home/edgar/DevOps/ML-Argo-CD/model-train/data", 
    train=False,
    transform=transform,
    download=True
)

image, label = dataset[321]
pixels = image.numpy().flatten().tolist()

#response = requests.post("http://localhost:8000/predict", json={"pixels": pixels})
response = requests.post("http://mnist-api.local/predict", json={"pixels": pixels})


print(f"Status: {response.status_code}")
#print(f"Response: {response.text}")

result = response.json()
print(f"Реальная цифра:       {label}")
print(f"Предсказание модели:  {result['predicted_class']}")
print(f"Уверенность:          {result['confidence']:.2%}")
