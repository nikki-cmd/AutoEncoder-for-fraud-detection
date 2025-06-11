import torch
import numpy as np
import configs.model_config as model_cfg
import configs.data_config as data_cfg
from autoencoder.trainer import Autoencoder
from dataloaders.SplitData import SplitData
from dataloaders.dataloader import Dataloader
from tools.preprocessing import Preprocessing, Tensors
from tools.metrics import Evaluation, Graphics

model_path = model_cfg.test_model_path
data_path = data_cfg.DATA_PATH
input_dim = model_cfg.input_dim
labels = data_cfg.LABELS

def load_validation_data():
    data = Dataloader().__load__()

    standart = Preprocessing(data)
    data = standart.__standardize__()

    split_data = SplitData(data)
    X_train, X_validation, X_test, y_validation, y_test = split_data.__split__()

    tensor = Tensors(X_train, X_validation, X_test)
    X_train, X_validation, X_test = tensor.__to_tensor__()
    
    return X_validation, y_validation

def load_test_data():
    data = Dataloader().__load__()

    standart = Preprocessing(data)
    data = standart.__standardize__()

    split_data = SplitData(data)
    X_train, X_validation, X_test, y_validation, y_test = split_data.__split__()

    tensor = Tensors(X_train, X_validation, X_test)
    X_train, X_validation, X_test = tensor.__to_tensor__()
    
    return X_test, y_test

def validate_AE():
    input_data, targets = load_validation_data()
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        validation_predictions = model(input_data)

    anomaly_scores = np.mean((input_data.numpy() - validation_predictions.numpy()) ** 2, axis=1)
    
    return anomaly_scores, targets

def test_AE():
    input_data, targets = load_test_data()
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        test_predictions = model(input_data)

    anomaly_scores = np.mean((input_data.numpy() - test_predictions.numpy()) ** 2, axis=1)
    
    return anomaly_scores, targets


#anomaly_scores, targets = validate_AE()

anomaly_scores, targets = test_AE()

    
T = np.percentile(anomaly_scores, model_cfg.treshold)

predicted_classes = np.array((anomaly_scores > T).astype(int))
ev = Evaluation(predicted_classes, targets)

ev.__confusion_matrix__()
ev.__precision_score__()
ev.__recall_score__()
ev.__accuracy_score__()
ev.__f1_score__()
print('Treshold:', model_cfg.treshold)

gr = Graphics(anomaly_scores, targets, predicted_classes)
gr.__confusion_matrix__()
gr.__roc_curve__()
gr.__pr_curve__()

'''per = 99.0
p_history = []
pres_history = []
tn_hist = []
fp_hist = []

for _ in range(0, 9):
    p_history.append(per) 
    T = np.percentile(anomaly_scores, per)
    predicted_classes = np.array((anomaly_scores > T).astype(int))
    ev = Evaluation(predicted_classes, y_validation)
    pres_history.append(ev.__precision_score__())
    tn, tp, fp, fn = ev.__classification_metrix__()
    tn_hist.append(tn)
    fp_hist.append(fp)
    per+=0.1

plt.plot(p_history, pres_history, color="red")
plt.ylabel('presicion')
plt.xlabel('treshold')
plt.show()
plt.plot(p_history, tn_hist, color="blue")
plt.ylabel('presicion')
plt.xlabel('treshold')
plt.show()
plt.plot(p_history, fp_hist, color="green")
plt.ylabel('presicion')
plt.xlabel('treshold')
plt.show()
'''




