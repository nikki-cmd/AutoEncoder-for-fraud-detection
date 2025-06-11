import torch
import torch.nn as nn
import configs.model_config as model_cfg

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = model_cfg.encoder
        self.decoder = model_cfg.decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


