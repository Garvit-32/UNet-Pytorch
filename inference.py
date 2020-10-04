import torch
from torchvision import transforms

from UNet import UNet

import cv2
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

img = cv2.imread('test_img/5.jpg')
img = cv2.resize(img,(160,160))

img = transform(img)

img = torch.unsqueeze(img,0)
img = img.cuda()

model = torch.load("checkpoints/fcn_model_100.pt")
model = model.cuda()

y = model(img)

y = torch.squeeze(y)
y = y.detach().cpu().numpy()

y = y.swapaxes(0,2).swapaxes(0,1)

y = y[:,:,0]

cv2.imshow('prediction',y)
cv2.waitKey(0)
