import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class customer_Dataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data'))

    def __getitem__(self, idx):
        img_name = os.listdir('data')[idx]
        imgA = cv2.imread('data/' + img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('data_msk/' + (os.path.splitext(img_name)[0] + '.tif'), -1)

        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)

        # imgB = cv2.imreadmulti('data_msk/' + (os.path.splitext(img_name)[0] + '.tif'), None, flags=-1)
        # buf = []
        # for idx in range(len(imgB[1])):
        #     # print(imgB[1][:][idx].shape)
        #     img = cv2.resize(imgB[1][:][idx], (160, 160))
        #     img = img / 255
        #     img = img.astype('uint8')
        #     img = onehot(img, 2)
        #     img = img.transpose(2, 0, 1)
        #     img = torch.FloatTensor(img)
        #     buf.append(img)

        if self.transform:
            imgA = self.transform(imgA)

        # return imgA, buf
        return imgA, imgB



if __name__ == '__main__':
    data = customer_Dataset(transform)

    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size

    train_dataset, test_dataset = train_test_split(data, test_size=0.3, random_state=42)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=16)

    for train_batch in train_dataloader:
        pass

    # for test_batch in test_dataloader:
    #     print(test_batch)
