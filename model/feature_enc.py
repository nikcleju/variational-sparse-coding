import numpy as np
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):

    def __init__(self, img_size):
        super(MLPEncoder, self).__init__()
        self.enc = nn.Sequential(
                nn.Linear(img_size**2, (img_size**2) * 2),
                nn.ReLU(),
                nn.Linear((img_size**2) * 2, (img_size**2) * 4),
                nn.ReLU(),
                nn.Linear((img_size**2) * 4, (img_size**2) * 2),
                nn.ReLU(),
                nn.Linear((img_size**2) * 2, (img_size**2)),
                nn.ReLU())    

    def forward(self, x):
        return self.enc(x)

class ConvEncoder(nn.Module):

    def __init__(self, num_feat, num_channels, im_size=64):
        super(ConvEncoder, self).__init__()
        if im_size == 64:
            self.enc = nn.Sequential(
                    # 32 x 32
                    nn.Conv2d(num_channels, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # 16 x 16
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # 8 x 8
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    # 4 x 4
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    # 1 x 1
                    nn.Conv2d(256, 256, 4, 2, 0),
                    nn.BatchNorm2d(256),
                    nn.ReLU())
        else:
            self.enc = nn.Sequential(
                    # 14 x 14
                    nn.Conv2d(num_channels, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # 7 x 7
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # 4 x 4
                    nn.Conv2d(64, 128, 4, 1, 0),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    # 1 x 1
                    nn.Conv2d(128, 128, 4, 2, 0),
                    nn.BatchNorm2d(128),
                    nn.ReLU()               
            )
        #self.linear = nn.Linear(256, num_feat)
        self.linear = nn.Identity()


    def forward(self, x):
        z = self.enc(x)
        z = z.reshape(len(x), -1)
        return self.linear(z)

# Nic
class ConvEncoder2d(nn.Module):

    def __init__(self, channels_seq, kernel_size=(5,5)):
        super(ConvEncoder2d, self).__init__()

        self.enc = nn.Sequential()
        for out_channels in channels_seq:
            self.enc.append(nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=(1,1), padding='same'))
            self.enc.append(nn.BatchNorm2d(out_channels))
            self.enc.append(nn.ReLU())

    def forward(self, x):
        return self.enc(x)

class ConvDecoder(nn.Module):
    def __init__(self, num_feat, num_channels, im_size=64):
        super(ConvDecoder, self).__init__()
        self.im_size = im_size
        if im_size == 64:
            self.linear = nn.Linear(num_feat, 256*2*2)
            self.dec = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, num_channels, 4, 2, 1),
                    nn.Sigmoid()
            )
        elif im_size == 28:
            self.linear = nn.Linear(num_feat, 128)
            self.dec = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, 4, 2, 0),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 1, 0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, num_channels, 4, 2, 1),
                    nn.Sigmoid()
            )

    def forward(self, x):
        # If decoding multiple samples per image, handle that across batches
        if x.dim() > 2:
            if self.im_size == 64:
                x_hat = self.linear(x).reshape((len(x) * x.shape[1]), -1, 2, 2)
            elif self.im_size == 28:
                x_hat = self.linear(x).reshape((len(x) * x.shape[1]), -1, 1, 1)
            x_hat = self.dec(x_hat)
            return x_hat.reshape(len(x), x.shape[1], *x_hat.shape[1:])
        else:
            if self.im_size == 64:
                x_hat = self.linear(x).reshape(len(x), -1, 2, 2)
            elif self.im_size == 28:
                x_hat = self.linear(x).reshape(len(x), -1, 1, 1)
            return self.dec(x_hat)

# Nic
class ConvDictDecoder(nn.Module):

    def __init__(self, n_channels, out_channels=1, kernel_size=(5,5), stride=(1,1), padding='same'):
        super(ConvDictDecoder, self).__init__()
        self.depthconv =  nn.Conv2d(in_channels = n_channels, 
                            out_channels = n_channels,
                            kernel_size=kernel_size, 
                            stride=stride,
                            groups=n_channels, 
                            padding=padding, 
                            bias=False)
        # No bias ... Explain all

        self.shape = self.depthconv.weight.shape

    def forward(self, x):
        # x.shape = [1, 5, 32, 512, 512]
        # y.shape = [1, 5,  1, 512, 512]
        if len(x.shape) == 5:
            #y = torch.cat( self.depthconv(x[:,i,:,:,:]).sum(dim=0) for i in range(x.shape[1]))
            y = torch.stack([self.depthconv(slice.squeeze(dim=1)).sum(dim=1, keepdim=True) for slice in torch.split(x, 1, dim=1)], dim=1)
        elif len(x.shape) == 4 :
            y = self.depthconv(x).sum(dim=1, keepdim=True)
        
        return y
        #return self.depthconv(x).sum(dim=0)
        
    # def forward_good(self, x):
    #     # x.shape = [1, 5, 32, 512, 512]
    #     # y.shape = [1, 5,  1, 512, 512]
    #     if len(x.shape) == 5:
    #         #y = torch.cat( self.depthconv(x[:,i,:,:,:]).sum(dim=0) for i in range(x.shape[1]))
    #         y = torch.stack((self.depthconv(slice.squeeze(dim=1)).sum(dim=1, keepdim=True) for slice in torch.split(x, 1, dim=1)), dim=1)
    #     elif len(x.shape) == 4 :
    #         y = self.depthconv(x).sum(dim=1, keepdim=True)
        
    #     return y
