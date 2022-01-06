from model import *
from config import *
from argparse import *
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import Resize


parser = ArgumentParser()
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--data_path", type=str)
args = parser.parse_args()


def calc_acc(preds, labels):
    _, pred = torch.max(preds, 1)
    acc = torch.sum(pred == labels.data, dtype=torch.float64) / len(preds)
    return acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = torchvision.datasets.ImageFolder(
    root=args.data_path, transform=transform)
train_data = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_loss = 0.0
    train_acc = 0.0

    for images, labels in tqdm(train_data):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss = loss_function(preds, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss
        train_acc += calc_acc(preds, labels)

    total_loss = train_loss / len(train_data)
    total_acc = train_acc / len(train_data)
    print(f"Epoch: {epoch+1}, Loss: {total_loss}, Accuracy: {total_acc}")
