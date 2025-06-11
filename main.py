from dataloaders.SplitData import SplitData
from dataloaders.dataloader import Dataloader
from tools.preprocessing import Preprocessing, Tensors
from tools.weights import Weights
from autoencoder.trainer import Autoencoder
import torch.nn as nn
import torch
import configs.model_config as model_cfg
from torch.utils.data import DataLoader, TensorDataset
from tools.metrics import Loss

data = Dataloader().__load__()

standart = Preprocessing(data)
data = standart.__standardize__()
#data = standart.__normalize__()

split_data = SplitData(data)
X_train, X_validation, X_test, y_validation, y_test = split_data.__split__()

tensor = Tensors(X_train, X_validation, X_test)
X_train, X_validation, X_test = tensor.__to_tensor__()

train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=model_cfg.batch_size, shuffle=True)

model = Autoencoder()

model.apply(Weights.init_weights)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.learning_rate)

# Обучение модели
history = []
model.train()
for epoch in range(model_cfg.epochs):
    epoch_loss = 0
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    history.append(epoch_loss / len(train_loader))
    print(f'Epoch [{epoch + 1}/{model_cfg.epochs}], Loss: {epoch_loss / len(train_loader)}')

# Сохранение модели
torch.save(model.state_dict(), model_cfg.model_path)
print(f"Модель сохранена в {model_cfg.model_path}")

metrics = Loss(history=history)

metrics.__loss_score__()



