import torch 

class Preprocessing:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __standardize__(self):
        targets = self.dataset['Class']
        data = self.dataset.drop('Class', axis = 1)
        
        mean = data.mean(axis = 0)
        std = data.std(axis=0)

        data = (data - mean) / std        
        data['Class'] = targets
        return data
    
    def __normalize__(self):
        targets = self.dataset['Class']
        data = self.dataset.drop('Class', axis = 1)
        
        ma = data.max()
        mi = data.min()

        data = (data - mi) / (ma - mi)     
        data['Class'] = targets
        return data
    
class Tensors():
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

    def __to_tensor__(self):
        train = torch.tensor(self.train.values, dtype=torch.float32)
        validation = torch.tensor(self.validation.values, dtype=torch.float32)
        test = torch.tensor(self.test.values, dtype=torch.float32)
        return train, validation, test