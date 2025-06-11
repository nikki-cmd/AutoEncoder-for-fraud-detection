import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.nn as nn
from sklearn.metrics import precision_score

# Настройки
MODEL_PATH = "models/2025_03_12_15_01_05.pth"
DATA_PATH = "creditcard.csv"
LABELS = ["Normal", "Fraud"]

# Загрузка модели
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

# Загрузка данных
dataframe = pd.read_csv(DATA_PATH)
print(dataframe.columns)
#dataframe = dataframe[100000:]
features = [col for col in dataframe.columns if col != 'Class']
scaler = StandardScaler()
dataframe[features] = scaler.fit_transform(dataframe[features])

# Разделение данных на нормальные и мошеннические
normal_data = dataframe[dataframe['Class'] == 0].drop(['Class'], axis=1)
fraud_data = dataframe[dataframe['Class'] == 1].drop(['Class'], axis=1)

print(normal_data.shape)
print(fraud_data.shape)

# Объединение данных
test_x = np.concatenate([normal_data, fraud_data])
test_y = np.array([0] * len(normal_data) + [1] * len(fraud_data))

test_x = torch.tensor(test_x, dtype=torch.float32)

# Загрузка модели
input_dim = test_x.shape[1]
model = Autoencoder(input_dim=input_dim)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# Оценка модели
with torch.no_grad():
    test_predictions = model(test_x)
    mse = torch.mean((test_x - test_predictions) ** 2, dim=1).numpy()
    
print(test_predictions.size())

# Гистограмма Anomaly_score
'''plt.figure(figsize=(14, 8))
plt.bar(range(len(mse)), mse, color='purple')
plt.title("Anomaly Score (Reconstruction Error) for Transactions")
plt.xlabel("Transaction Index")
plt.ylabel("Reconstruction Error")
plt.show()'''

# Визуализация ошибок
plt.figure(figsize=(14, 8))
sns.histplot(mse[test_y == 0], bins=50, color='blue', label='Normal', kde=True)
sns.histplot(mse[test_y == 1], bins=50, color='red', label='Fraud', kde=True)
plt.legend()
plt.title('MSE Distribution for Normal and Fraudulent Transactions')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# ROC-кривая
fpr, tpr, _ = roc_curve(test_y, mse)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(14, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Матрица ошибок
threshold = np.percentile(mse, 99)  # Пример порога
pred_y = (mse > threshold).astype(int)
conf_matrix = confusion_matrix(test_y, pred_y)

precision = precision_score(test_y, pred_y)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Reds', xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix for Normal and Fraudulent Transactions\nPrecision: {precision:.4f}")
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Вывод precision
print(f"Precision: {precision:.4f}")