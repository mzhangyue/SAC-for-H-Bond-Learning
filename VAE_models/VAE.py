import torch
import torch.nn as nn
from VAE_base import VAE_base

class VAE(VAE_base):

    def init(self, input_dims, hidden_dims=8):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential (
            nn.Linear(input_dims, 16)
        )

        self.mlp1 = nn.Linear(16)
        self.mlp2 = nn.Linear()

        self.decoder = nn.Sequential (

        )

    def encode(self):
        return

    def decode(self):
        return


    