import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        self.fc_0 = nn.Linear(1, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        self.actvn = F.relu

        # self.conv_in = nn.Conv3d(1, 32, 3, padding=1)
        #
        # self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        # self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        # self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        # self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        # self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, p, x, c=None, **kwargs):
        batch_size, T, D = p.size()
        # output size: B x T X F

        net = self.fc_0(x.unsqueeze(-1))
        net = net + self.fc_pos(p)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce
        #  to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)


        # print("Forward is called -----------", mean, logstd)
        return mean, logstd


class VoxelEncoder(nn.Module):
    ''' Latent encoder class.


    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        # self.fc_pos = nn.Linear(dim, 128)

        # if c_dim != 0:
        #     self.fc_c = nn.Linear(c_dim, 128)

        # self.fc_0 = nn.Linear(1, 128)
        # self.fc_1 = nn.Linear(128, 128)
        # self.fc_2 = nn.Linear(256, 128)
        # self.fc_3 = nn.Linear(256, 128)

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)
        # self.conv_in2 = nn.Conv3d(16, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(16, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 , 128)
        self.actvn = F.relu
        # self.fc = nn.Linear(128, z_dim)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)



    def forward(self, p, x, c=None, **kwargs):
        batch_size = c.size(0)

        c = c.unsqueeze(1)
        # print(c.double())
        net = self.conv_in(c)
        # net = self.conv_in2(c)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        # print("Shape: ", net.shape)
        hidden = net.view(batch_size, 512)
        c_out = self.fc((hidden))
        c_out = self.actvn(c_out)
        mean = self.fc_mean(c_out)
        logstd = self.fc_logstd(c_out)


        # print("Forward is called -----------", mean, logstd)
        return mean, logstd
