import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Loss:
    def __init__(self, history):
        self.loss_history = history
        
    
    def __loss_score__(self):
        plt.plot(self.loss_history)
        plt.ylabel('Loss Score')
        plt.xlabel('Epoch')
        plt.show()
    
class Evaluation():
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
    def __accuracy_score__(self):
        tn, tp, fp, fn = self.__classification_metrics__()
        print('Accuracy:', (tp+tn)/(tp+tn+fp+fn))
    
    def __confusion_matrix__(self):
        tn, tp, fp, fn = self.__classification_metrics__()
        
        print("Confusion Matrix:")
        print(f"                Predicted 0   Predicted 1")
        print(f"Actual 0         {tn}    {fp}")
        print(f"Actual 1         {fn}    {tp}")
        print()
        print(f'Found {tp}, out of {tp+fn}')
    
    def __precision_score__(self):
        tn, tp, fp, fn = self.__classification_metrics__()
        print('Precission:', tp/(tp+fp))
    
    def __recall_score__(self):
        tn, tp, fp, fn = self.__classification_metrics__()
        print('Recall:', tp/(tp+fn))
    
    def __f1_score__(self):
        tn, tp, fp, fn = self.__classification_metrics__()
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        print('F1-score:', 2 * ((p*r)/(p+r)))
    
    def __classification_metrics__(self):
        y_true_values = self.y_true.to_numpy()
        tn = 0
        tp = 0
        fp = 0
        fn = 0

        for i in range(len(self.y_pred)):
            score = self.y_pred[i]
            true_value = y_true_values[i]
            if score == 0 and true_value == 0:
                tn += 1
                
            elif score == 1 and true_value == 1:
                tp += 1
                
            elif score == 1 and true_value == 0:
                fp += 1
                
            elif score == 0 and true_value == 1:
                fn += 1
                
        return tn, tp, fp, fn

class Graphics():
    def __init__(self, anomaly_scores, y_true, y_pred):
        self.y_scores = anomaly_scores
        self.y_true = y_true
        self.y_pred = y_pred
    
    def __roc_curve__(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        auc = roc_auc_score(self.y_true, self.y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC-curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    def __pr_curve__(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        ap = average_precision_score(self.y_true, self.y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR-curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    def __confusion_matrix__(self):
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
                