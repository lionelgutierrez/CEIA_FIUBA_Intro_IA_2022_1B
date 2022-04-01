import numpy as np
from ejercicio6 import distanciaCentroides

def indicesMinDist(distCent:np.array):
    '''Para cada fila de la matriz distCent, la funcion devuelve el índice de la fila con distancia euclídea más pequeña.
       La distancia se obtiene desde la funcion definida en el ejercicio6 distanciaCentroides(X,C) 
    '''
    return np.argmin(distCent,axis=0)


#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    X = np.array([[ 1.,2.,3.], [ 4.,5.,6.] , [7.,8.,9.] ])
    c = np.array([[ 1.,0.,0.] , [0.,1.,1.] ])
    print("Matriz de puntos: \n",X)
    print("Centorides: \n",c)
    distancia = distanciaCentroides(X,c) 
    print("Distancia calculada:\n ",distancia)    
    indiMinD = indicesMinDist(distancia)
    print("Indices de centroide por punto obtenido con la distancia: ",indiMinD)
