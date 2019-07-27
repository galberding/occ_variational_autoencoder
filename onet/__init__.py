import torch
import torch.nn as nn
from torch import distributions as dist
from onet import encoder_latent, decoder
from onet.encoder_latent import VoxelEncoder, Encoder
from scipy.stats import norm
# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
    'advance': encoder_latent.VoxelEncoder
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.
    As default the network will use the Voxel encoder combined with the simple decoder.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        z_dim: dimension of latent space
        device (device): torch device
    '''

    def __init__(self, decoder=decoder_dict["simple"], encoder=None, encoder_latent=encoder_latent_dict["advance"], p0_z=None,
                 device=None, z_dim=2):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(
                torch.zeros(z_dim, device=device),
                torch.ones(z_dim, device=device)
            )

        self.decoder = decoder(z_dim=z_dim).to(device)

        if encoder_latent is not None:
            if isinstance(encoder_latent, VoxelEncoder):
                print("Voxel encoder chooosed!")
                self.latent_voxel = True
            else:
                self.latent_voxel = False
            self.encoder_latent = encoder_latent(z_dim=z_dim).to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z
        self.counter = -5.0
        self.dim = 0


    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        z = self.get_z_from_prior((batch_size,), sample=sample)


        # print("Sample z ")


        p_r = self.decode(p, z, c, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''

        print("elbo")
        c = self.encode_inputs(inputs)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        # print("Compute Z: ", z)
        p_r = self.decode(p, z, c, **kwargs)
        # print("Occupancys: ", occ, "Point: ",p)
        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        # print("Reconstructiopn Error", rec_error)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        elif self.latent_voxel:
            c = inputs
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, z, c, **kwargs)
        # print(logits)
        p_r = dist.Bernoulli(logits=logits)
        # print(p_r)
        return p_r

    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
            # print("Mean: ",mean_z,"\nstd: ", logstd_z)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)
        # print("Occupancys during inferance: ", occ)
        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        # print("Sampled Z: ", z)

        # if self.dim > 9:
        #     exit(42)
        # z = torch.tensor([[0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        # z[0,self.dim] = self.counter
        # # print(z)
        # self.counter += 0.5
        # if self.counter > 5.0:
        #     self.counter = -5.0
        #     self.dim += 1

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
