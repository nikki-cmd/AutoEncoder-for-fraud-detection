import torch.nn as nn
import torch.nn.init as init

class Weights:
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.zeros_(m.bias)             