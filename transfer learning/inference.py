from PIL import Image
import numpy as np
import cv2
import argparse
from model import *
import torch
from torchvision import transforms
import torchvision

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device', default='cpu', type=str)
my_parser.add_argument('--weight', type=str)
my_parser.add_argument('--image_path', type=str)
args = my_parser.parse_args()

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device('cuda' if torch.cuda.is_available()
                      and args.device == 'GPU' else 'cpu')
model = Model().to(device)

model.load_state_dict(torch.load(
    args.model_path, map_location=torch.device(args.device)))

model.eval()
image = cv2.imread(args.image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (32, 32))
PIL_image = Image.fromarray(image)
tensor = transform(PIL_image).unsqueeze(0).to(device)
pred = model(tensor)
pred = pred.cpu().detach().numpy()
pred = np.argmax(pred)
