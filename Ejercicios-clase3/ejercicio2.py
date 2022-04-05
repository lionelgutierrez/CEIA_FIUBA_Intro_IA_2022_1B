import numpy as np


def removerNans(dataset):
    '''Dado un dataset, le quita todas las filas y columnas que tengan algun nan
       Devuelve el dataset sin las filas y columnas con nans
    '''    
    filtroColumnas = ~(np.isnan(dataset).any(axis=0))
    filtroFilas = ~(np.isnan(dataset).any(axis=1))
    dataset = dataset[filtroFilas]
    dataset = dataset[:,filtroColumnas]
    return dataset

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    n = 10
    m = 5
    dataset = np.arange(n*m,dtype="float").reshape(n,m)    
    dataset[0][0] = np.nan
    dataset[3][2] = np.nan
    print("Dataset original:\n",dataset)
    dataset_sin_nans = removerNans(dataset)
    print("\nDataset sin filas ni columnas con nans:\n",dataset_sin_nans)
