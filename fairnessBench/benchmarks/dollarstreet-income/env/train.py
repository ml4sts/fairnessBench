import os
import ast
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DollarStreetDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        self.data['imagenet_sysnet_id'] = self.data['imagenet_sysnet_id'].apply(ast.literal_eval)
        all_synsets = [syn[0] for syn in self.data['imagenet_sysnet_id']]
        unique_synsets = sorted(set(all_synsets))
        self.synset2idx = {syn: idx for idx, syn in enumerate(unique_synsets)}
        self.data['label'] = self.data['imagenet_sysnet_id'].apply(lambda x: self.synset2idx[x[0]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['imageRelPath'])
        image = Image.open(img_path).convert("RGB")
        label = row['label']
        income = row['income']
        if self.transform:
            image = self.transform(image)
        return image, label, income

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_csv = "./images_v2_imagenet_train.csv"
test_csv = "./images_v2_imagenet_test.csv"

train_dataset = DollarStreetDataset(csv_path=train_csv, root_dir=".", transform=transform) 
test_dataset = DollarStreetDataset(csv_path = test_csv, root_dir=".", transform=transform) 
num_classes = len(train_dataset.synset2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 5

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct/total

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.2f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

model.eval()
submission = pd.DataFrame(columns=list(range(num_classes)), index=range(len(test_dataset)))
for idx, (image, label, income) in enumerate(test_dataset):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        pred = model(image)
        pred_prob = torch.softmax(pred[0], dim=0)
        submission.loc[idx] = pred_prob.cpu().tolist()
submission.to_csv("submission.csv", index=False)

