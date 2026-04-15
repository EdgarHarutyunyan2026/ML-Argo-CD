import requests
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="/home/edgar/DevOps/ML-Argo-CD/model-train/data/",
    train=False,
    transform=transform,
    download=True
)

image, label = dataset[32]
pixels = image.numpy().flatten().tolist()

url = "http://mnist-api.local/predict"
headers = {"Host": "mnist-api.local"} 

response = requests.post(url, headers=headers, json={"pixels": pixels})

print(f"Status: {response.status_code}")
try:
    result = response.json()
    print(f"Реальная цифра:       {label}")
    print(f"Предсказание модели:  {result['predicted_class']}")
    print(f"Уверенность:          {result['confidence']:.2%}")
except Exception as e:
    print("Ошибка при разборе JSON:", e)
    print("Ответ сервера:", response.text)
