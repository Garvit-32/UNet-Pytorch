import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomDataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from UNet import UNet
import pandas as pd
# import wandb
# import visdom

# wandb.init(project="unet")


train_data_dir = 'data'
train_mask_dir = 'mask'

train_df = pd.read_csv('train.csv')


if __name__ == '__main__':

    # vis = visdom.Visdom()
    train = CustomDataset(train_df,train_data_dir,train_mask_dir)


    net = UNet(n_channels=3,n_classes=2)

    net = net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr = 1e-3,momentum=0.9)

    train_loader = DataLoader(train,batch_size=4,shuffle=True,num_workers=8)


    for epoch in range(100):
        
        index = 0 
        epoch_loss = 0
        

        for item in train_loader:
            index += 1
            img = item['img']
            true_mask = item['mask']

            img = torch.autograd.Variable(img)
            true_mask = torch.autograd.Variable(true_mask)

            img = img.cuda()
            true_mask = true_mask.cuda()
            optimizer.zero_grad()
            output = net(img)
            output = torch.sigmoid(output)
            loss = criterion(output ,true_mask)
            loss.backward()
            iter_loss = loss.item()
            epoch_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().data.numpy().copy()
            output_np = np.argmin(output_np, axis=1)

            true_mask_np = true_mask.cpu().data.numpy().copy()
            true_mask_np = np.argmin(true_mask_np,axis=1)

            # wandb.log({'epoch':epoch,'loss':iter_loss})

            if index % 20 == 0:
                print('epoch {},  {}/{} , loss is {}'.format(epoch,index,len(train_loader),iter_loss))
                # vis.close()
                # vis.images(output_np[:,None,:,:],opts=dict(title="pred-epoch {}".format(epoch)))
                # vis.images(true_mask_np[:,None,:,:],opts=dict(title='label epoch {}'.format(epoch)))

        # wandb.log({'epoch loss':epoch_loss/len(train_loader),'accuracy':1-(epoch_loss/len(train_loader))})
        print('epoch loss = %f'%(epoch_loss/len(train_loader)))
        
        if (epoch+1) % 5 ==0:
            torch.save(net, 'checkpoints/fcn_model_{}.pt'.format(epoch+1))
            print('saving checkpoints/fcn_model_{}.pt'.format(epoch+1))



