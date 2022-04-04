import numpy as np
import sys
import scipy.stats as sps

X[1][2] = np.nan
X[9][4]= np.nan
X[8][1]= np.nan
print(X)

def reemplazaNanPorMedia(dataset):
    mediaColumnas = np.nanmean(dataset,axis=0,keepdims=True) 
    print(mediaColumnas)
    print(np.nan_to_num(X,nan=mediaColumnas))
    
reemplazaNanPorMedia(X)