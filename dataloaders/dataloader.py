import pandas as pd
from configs import data_config

class Dataloader:
    def __init__(self):
        self.path = data_config.DATA_PATH
        self.dataframe_size = data_config.dataframe_size
    
    def __load__(self):
        df = pd.read_csv(self.path)
        
        #df = df[:self.dataframe_size]
        self.df = df
        return df
    
    def __get_dims__(self):
        df = pd.DataFrame()
        df = self.__load__()
        return df.shape[1] - 1 #cuz we delete Class column