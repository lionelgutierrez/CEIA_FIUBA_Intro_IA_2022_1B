import numpy as np
from ejercicio1 import norma_l2 

def ordenarMatriz(matriz:np.array):
    '''Permite obtener la matriz original ordenada por fila según la norma l2.'''
    normas = norma_l2(matriz)
    normasOrden = np.argsort(normas * -1)
    return matriz[normasOrden,:]
 
#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    a4 = np.array([[ 100  ,-1 , -2,  -3], [ -4 , -8 , -6 , 5] , [-8  ,-99 ,-10 ,-11] ,[-12 ,-13 ,15, -16]])
    print("Matriz base: \n",a4)
    print("Norma L2 de matriz: ",norma_l2(a4))
    print("Matriz ordeneada: \n",ordenarMatriz(a4))