import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

#TODO: reparametrization trick for backpropagating through sampling process. do i understand it? Done
#TODO: what additional loss term are we adding to AE to make it a VAE?
#TODO: the dimensions of latent space are all independent. could use a multi-variate gaussian with a covariance matrix to model this.

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1 : Fill in self.convs following the given architecture
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        self.encoder_block = nn.Sequential(*[nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), 
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(), 
                                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)])
        fc_input = 256 * (input_shape[1]//8) * (input_shape[2]//8)
        self.fc = nn.Linear(fc_input, self.latent_dim, bias=True) #TODO: sum across pixels, or flatten everything. How to decide?

    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim

        # x = self.encoder_block(x).sum(dim=[2,3]) #TODO: sum across pixels, or flatten everything. How to decide?
        x = self.encoder_block(x)
        # print("in Encoder fwd", x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x



class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim

        self.encoder_block = nn.Sequential(*[nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), 
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(), 
                                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)])
        fc_input = 256 * (input_shape[1]//8) * (input_shape[2]//8)
        self.fc = nn.Linear(fc_input, 2*self.latent_dim, bias=True) #TODO: sum across pixels, or flatten everything. How to decide?




    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std

        x = self.encoder_block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        mu = x[:,:self.latent_dim]
        log_std = x[:,self.latent_dim:]

        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1: fill in self.base_size
        self.base_size = [256 , (output_shape[1]//8) , (output_shape[2]//8)]
        # print("In Decoder init: ", latent_dim, self.base_size, torch.prod(self.base_size))
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))

        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.decoder_block = nn.Sequential(*[nn.ReLU(),
                                            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])

    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs.
        #TODO: self.fc what?
        # print("In decoder: fix shape", z.shape)
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.decoder_block(out)
        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
