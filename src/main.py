import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import json
from sklearn import metrics
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True, )

efficientnet.eval().to(device)

class ImageNet(torch.utils.data.Dataset):
  def __init__(self, images_path, label_path):
    self.images_path = images_path
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    self.labels = json.load(open(label_path))
    self.image_files = [f for f in os.listdir(images_path) if f.endswith('.JPEG')]
  
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    image_path = os.path.join(self.images_path, self.image_files[idx])
    image = Image.open(image_path).convert('RGB')
    image = self.transform(image)
    return image, self.labels[self.image_files[idx]]

dataset = ImageNet(images_path='dataset', label_path='labels.json')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
target_names = [v for v in json.load(open('categories.json')).values()]

targets = []
predictions = []
start = time.time()
for i, (images, labels) in enumerate(dataloader):
  images = images.to(device)
  logits = efficientnet(images)
  preds = torch.argmax(logits, axis=1)
  targets.extend(labels.numpy().tolist())
  predictions.extend(preds.cpu().numpy().tolist())
  print(f'\r{i+1}/{len(dataloader)}', end='')
print(f'\nInference time: {time.time() - start}')
print(f"Time per image: {(time.time() - start)/len(dataloader)}")

report = metrics.classification_report(targets, predictions, target_names=target_names, labels=range(len(target_names)))
with open('report.txt', 'w') as f:
  f.write(report)
  f.write(f"\nTime per image: {(time.time() - start)/len(dataloader)}")
