import numpy as np

def reemplazaNanPorMedia(dataset):
    '''Dado un dataset, reemplaza los nans de cada columna por la media de la misma
       Devuelve el dataset con los nans reemplazados segun el criterio indicado
    '''     
    mediaColumnas = np.nanmean(dataset,axis=0,keepdims=True) 
    return np.nan_to_num(dataset,nan=mediaColumnas)

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    n = 10
    m = 5
    dataset = np.arange(n*m,dtype="float").reshape(n,m)    
    dataset[1][2] = np.nan
    dataset[9][4]= np.nan
    dataset[8][1]= np.nan
    print("Dataset original:\n",dataset)
    dataset_sin_nans = reemplazaNanPorMedia(dataset)
    print("\n Dataset con nans reemplazados por media:\n",dataset_sin_nans)    

