import requests
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    #root="/home/edgar/DevOps/MLOps/k8s/pytorch/model/data/",
    root="/home/edgar/DevOps/ML-api/Argo-CD/model-train/data/",
    train=False,
    transform=transform,
    download=True
)

image, label = dataset[37]
pixels = image.numpy().flatten().tolist()

#url = "http://192.168.49.2/predict"
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
