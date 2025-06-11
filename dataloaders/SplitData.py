from configs import data_config
import pandas as pd

class SplitData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_size = data_config.train_size
        self.validation_size = data_config.validation_size
        self.test_size = data_config.test_size
        self.random_state = data_config.RANDOM_SEED
        self.dataframe_len = data_config.dataframe_size
        
    
    def __get_labels__(self, dataframe):
        return dataframe['Class']
    
    def __clear_train_set__(self, df, train_set):
        fraud_trans = train_set[train_set['Class'] == 1]
        train_set = train_set[train_set['Class'] == 0]
        train_set = train_set.drop(['Class'], axis = 1)
        
        df = pd.concat([df, fraud_trans], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        return df, train_set
    
    '''def __split__(self):
        df = self.dataset
        #k = self.dataframe_len
        k = df.shape[0]
        
        num_samples_train = int(k * self.train_size)
        num_samples_validation = int(k * self.validation_size)
        num_samples_test = int(k * self.test_size)
        
        train_set = df[:num_samples_train]
        
        df, train_set = self.__clear_train_set__(df, train_set)
        
        validation_set = df[:num_samples_validation]
        
        test_set = df[num_samples_validation:num_samples_validation + num_samples_test]
        
        validation_labels = self.__get_labels__(validation_set)
        test_labels = self.__get_labels__(test_set)
        
        validation_set = validation_set.drop(['Class'], axis=1)
        test_set = test_set.drop(['Class'], axis=1)

        return train_set, validation_set, test_set, validation_labels, test_labels'''
    
    def __split__(self):
        df = self.dataset
        k = df.shape[0]
        
        # Вычисляем границы выборок
        train_end = int(k * self.train_size)
        validation_end = train_end + int(k * self.validation_size)
        
        # Разбиваем данные
        train_set = df[:train_end]
        validation_set = df[train_end:validation_end]
        test_set = df[validation_end:validation_end + int(k * self.test_size)]
        
        # Получаем метки и удаляем столбец с классами из признаков
        train_labels = self.__get_labels__(train_set)
        validation_labels = self.__get_labels__(validation_set)
        test_labels = self.__get_labels__(test_set)
        
        train_set = train_set.drop(['Class'], axis=1)
        validation_set = validation_set.drop(['Class'], axis=1)
        test_set = test_set.drop(['Class'], axis=1)

        return train_set, validation_set, test_set, validation_labels, test_labels
        
    