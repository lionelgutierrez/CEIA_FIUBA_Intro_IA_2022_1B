import numpy as np
import sys
import scipy.stats as sps

X[0][0] = np.nan
X[3][2] = np.nan
print(X)

def removerNans(dataset):
    filtroColumnas = ~(np.isnan(dataset).any(axis=0))
    filtroFilas = ~(np.isnan(dataset).any(axis=1))
    dataset = dataset[filtroFilas]
    dataset = dataset[:,filtroColumnas]
    return dataset

print(removerNans(X))