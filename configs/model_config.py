from dataloaders.dataloader import Dataloader
import torch.nn as nn
from datetime import datetime

dataloader = Dataloader()

learning_rate = 1e-3
batch_size = 128
input_dim = dataloader.__get_dims__()
epochs = 150

treshold = 99.6

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
model_path = "models/" + current_time + ".pth"
#model_path = "models/xa_init.pth"

test_model_path = "models/2025_05_26_09_27.pth"


encoder = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU()
)
decoder = nn.Sequential(
    nn.Linear(16, 32),  
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, input_dim)
)
print(f'input dim:{input_dim}')
'''encoder = nn.Sequential(
    nn.Linear(input_dim, 22),
    nn.ReLU(),
    nn.Linear(22, 15),
    nn.ReLU(),
    nn.Linear(15, 10),
    nn.ReLU(),
)
decoder = nn.Sequential(
    nn.Linear(10, 15),
    nn.ReLU(),
    nn.Linear(15, 22),
    nn.ReLU(),
    nn.Linear(22, input_dim)
)'''
