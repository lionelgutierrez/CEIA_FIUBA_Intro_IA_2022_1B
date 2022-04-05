import numpy as np

def zcore(X):
    '''Calcula el zcore de un dataset
       Recibe un dataset X y devulve el dataset aplicando la media a cada columna y dividiendo por el desvio standar 
    '''
    mediaColumnas = np.nanmean(X,axis=0,keepdims=True) 
    stdColumnas = np.nanstd(X,axis=0,keepdims=True)
    resultado = (X - mediaColumnas) / stdColumnas
    return resultado

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    n = 10
    m = 5
    dataset = np.arange(n*m,dtype="float").reshape(n,m)
    dataset_normalizado = zcore(dataset)    
    print("Dataset original:\n",dataset)
    print("\nDataset noralizado:\n",dataset_normalizado)

