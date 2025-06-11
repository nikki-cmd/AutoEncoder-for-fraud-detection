import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),  
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


LABELS = ['Normal', 'Fraud']
RANDOM_SEED = 27
LEARNING_RATE = 1e-3
MODEL_PATH = "fraud_autoencoder_with_anomalys.pth"
EPOCHS = 50

df = pd.read_csv('creditcard.csv')
df = df[:100000]

X_train, X_test = train_test_split(df, test_size = 0.2, random_state=RANDOM_SEED)
#X_train = X_train[X_train['Class'] == 0]

X_train = X_train.drop(['Class'], axis = 1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis = 1)

#standart.

mean = X_train.mean(axis = 0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(X_train)


print(np.mean(X_train, axis=0))
print(np.std(X_train, axis=0))
print(np.mean(X_test, axis=0))
print(np.std(X_test, axis=0))

batch_size = 100
orriginal_dim = 30
latent_dim = 5
intermediate_dim1 = 20
intermediate_dim2 = 10
epochs = 50
epsilon_std = 1.0

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.int64)


train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)

input_dim = X_train.shape[1]
model = Autoencoder(input_dim=input_dim)
model.apply(weights_init)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
history = []
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    history.append(epoch_loss / len(train_loader))
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}')

# Сохранение модели
torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена вoo {MODEL_PATH}")