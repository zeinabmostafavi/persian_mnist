
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
parser.add_argument("--weight", type=str)
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
test_data = torchvision.datasets.ImageFolder(
    root='/dataset', transform=transform)
test_data = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model = model.Model()
model = model.to(device)
model.load_state_dict(torch.load(args.wieght))
model.eval()

test_acc = 0.0

for image, labels in test_data:
    image = image.to(device)
    labels = labels.to(device)
    y_hat = model(image)
    test_acc += calc_acc(y_hat, labels)

test_acc = test_acc/len(test_data)

print(f"test acc: ", test_acc)
