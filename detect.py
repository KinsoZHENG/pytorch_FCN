from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Dataset_init import customer_Dataset, transform
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import cv2


def image_process_cv_to_vis(data):
    data = data.transpose(1, 2, 0)  # trans the shape from PIL to cv2

    b, g, r = cv2.split(data)   # get the b, g, r
    data = cv2.merge((r, g, b)) # due to the PIL shape as r g b,
    data = data.transpose(2, 0, 1)  # trans the shape from Cv2 tp PIL
    return data


def detect():
    """
    Initial Model, Load parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env='detect')
    vgg_model = VGGNet(requires_grad=True, show_params=False)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=5)
    fcn_model = torch.load('./checkpoints/fcn_model_100.pt')
    fcn_model.to(device).eval()

    """
    Initial test dataset
    """
    bag = BagDataset(transform)
    train_dataset, test_dataset = train_test_split(bag, test_size=0.3, random_state=42)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=16)


    with torch.no_grad():
        for index, (data, data_mask) in enumerate(test_dataloader):
            print("index: " + str(index))
            data = data.to(device)
            data_mask = data_mask.to(device)
            output = fcn_model(data)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            data = data.cpu().detach().numpy().copy()
            print(output_np[0].shape)
            image = data[0]
            image = image_process_cv_to_vis(image)

            for idx in range(len(data)):
                # data[idx] = image_process_cv_to_vis(data[idx])
                data[idx] = image_process_cv_to_vis(data[idx])

            data_mask_np = data_mask.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
            data_mask_np = np.argmin(data_mask_np, axis=1)
            vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
            vis.images(data_mask_np[:, None, :, :], win='test_label', opts=dict(title='label'))
            vis.images(data, win='input', opts=dict(title='input'))
            # vis.image(image, win='single', opts=dict(title='single'))
            # break


if __name__ == '__main__':
    detect()
