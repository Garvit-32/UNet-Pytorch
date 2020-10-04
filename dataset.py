from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
from onehot import onehot
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class CustomDataset(Dataset):

    def __init__(self,df,img_dir,mask_dir,transform=transform):
        self.transform = transform
        self.fname = df['0'].values.tolist()
        self.fname1 = df['1'].values.tolist()
        self.mask_dir = mask_dir
        self.img_dir = img_dir


    def __len__(self):
        return len(self.fname)

    def __getitem__(self,idx):
        img_path = self.fname[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img,(160,160))

        mask_path = self.fname1[idx]
        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask,(160,160))
        mask = mask/38
        mask = mask.astype(np.uint8)
        mask = onehot(mask,2)
        mask = mask.swapaxes(0,2).swapaxes(1,2)
        mask = torch.FloatTensor(mask)
        

        if self.transform:
            img = self.transform(img)


        item = {"img":img,'mask':mask}

        return item





