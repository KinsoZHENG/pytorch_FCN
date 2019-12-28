from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from apex import amp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Dataset_init import customer_Dataset, transform
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from detect import image_process_cv_to_vis


def train(epo_num=50, show_vgg_params=False):
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.9)
    lr = 0.01
    # fcn_model, optimizer = amp.initialize(fcn_model, optimizer, opt_level='O1', verbosity=0)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num + 1):
        # data set random init each epoch
        data = customer_Dataset(transform)
        train_dataset, test_dataset = train_test_split(data, test_size=0.1, random_state=1)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=16)

        train_loss = 0  # clear train as 0
        fcn_model.train()
        if (epo % 25 == 0) and (epo != 0):
            lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        """
        Part of training
        """
        for index, (data, data_msk) in enumerate(train_dataloader):
            # data.shape is torch.Size([4, 3, 160, 160])
            # data_msk.shape is torch.Size([4, 2, 160, 160])

            data = data.to(device)
            data_msk = data_msk.to(device)

            # data_msk = data_msk[0].to(device)
            # data_msk = data_msk[1].to(device)
            # data_msk = data_msk[2].to(device)
            # data_msk = data_msk[3].to(device)
            # data_msk = data_msk[4].to(device)

            optimizer.zero_grad()
            output = fcn_model(data)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, data_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            data_msk_np = data_msk.cpu().detach().numpy().copy()  # data_msk_np.shape = (4, 2, 160, 160)
            data_msk_np = np.argmin(data_msk_np, axis=1)
            data = data.cpu().detach().numpy().copy()

            if np.mod(index, 7) == 0:
                print('[train] epoch {}/{}, {}/{},\ttrain loss is {},\tlearning rate is {}'.format(
                    epo, epo_num, index, len(train_dataloader), iter_loss, lr))
                # vis.close()
                for idx in range(len(data)):
                    data[idx] = image_process_cv_to_vis(data[idx])

                vis.images(data,
                           win='train_input_ori_image',
                           opts=dict(title='train_input_ori_image'))

                vis.images(data_msk_np[:, None, :, :],
                           win='train_label',
                           opts=dict(title='train_label'))

                vis.images(output_np[:, None, :, :],
                           win='train_pred',
                           opts=dict(title='train prediction'))

                vis.line(all_train_iter_loss,
                         win='train_iter_loss',
                         opts=dict(title='train iter loss'))

        # vis.line(train_loss, epo, win='train_epoch_loss', opts=dict(title='train_epoch_loss'))

        """
        Part of eval
        """
        test_loss = 0  # clear test_loss
        fcn_model.eval()
        with torch.no_grad():
            for index, (data, data_msk) in enumerate(test_dataloader):

                data = data.to(device)
                data_msk = data_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(data)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, data_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                data_msk_np = data_msk.cpu().detach().numpy().copy()  # data_msk_np.shape = (4, 2, 160, 160)
                data_msk_np = np.argmin(data_msk_np, axis=1)
                data = data.cpu().detach().numpy().copy()

                if np.mod(index, 1) == 0:
                    print('[test]  epoch {}/{}, {}/{},\ttest  loss is {},\tlearning rate is {}'.format(
                        epo, epo_num, index, len(test_dataloader), iter_loss, lr))
                    # vis.close()
                    for idx in range(len(data)):
                        data[idx] = image_process_cv_to_vis(data[idx])

                    vis.images(data,
                               win='test_input_ori_image',
                               opts=dict(title='test_input_ori_image'))

                    vis.images(data_msk_np[:, None, :, :],
                               win='test_label',
                               opts=dict(title='test_label'))

                    vis.images(output_np[:, None, :, :],
                               win='test_pred',
                               opts=dict(title='test prediction'))

                    vis.line(all_test_iter_loss,
                             win='test_iter_loss',
                             opts=dict(title='test iter loss'))

        # vis.line(test_loss, epo, win='test_loss_loss', opts=dict(title='test_loss_loss'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s\n\n'
              % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt\n\n'.format(epo))


if __name__ == "__main__":
    train(epo_num=100, show_vgg_params=False)
