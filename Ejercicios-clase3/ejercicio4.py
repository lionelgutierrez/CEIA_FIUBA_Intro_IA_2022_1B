import numpy as np
import sys
import scipy.stats as sps

def separaDataset(dataset):
    ##dataset de n muestras
    #lo permuto en filas para no tener datos consecutivos que puedan tener relacion entre si
    #Divido en 70 train, 20 valid, 10 test
    
    train_set,valid_set,test_set = np.split(np.random.permutation(dataset),[int(0.7*len(dataset)),int(0.9*len(dataset))])
    return train_set,valid_set,test_set
    
separaDataset(X)  