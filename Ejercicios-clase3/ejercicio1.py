import numpy as np
import sys
import scipy.stats as sps

n = 10
m = 5
X = np.arange(n*m,dtype="float").reshape(n,m)#np.random.rand(n,m)

def zcore(X):
    mediaColumnas = np.nanmean(X,axis=0,keepdims=True) 
    stdColumnas = np.nanstd(X,axis=0,keepdims=True)
    #print(f"La media es {mediaColumnas} y la varianza es {stdColumnas}")
    
    #cada columna = columna - media / std
    #print("X: ",X)
    #print("media: ",mediaColumnas)
    #print("X - media: ",(X - mediaColumnas))
    resultado = (X - mediaColumnas) / stdColumnas
    return resultado
    #print("Resultado: ",resultado)
    
zcore(X)    