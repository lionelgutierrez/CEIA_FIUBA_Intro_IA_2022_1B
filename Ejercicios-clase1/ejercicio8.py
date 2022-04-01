import numpy as np
from ejercicio6 import distanciaCentroides 
from ejercicio7 import indicesMinDist


def seleccionarClusters():
    '''Funcion que permite al usaurio seleccionar el numero de clusters a utilizar para el calculo de kmenas'''
    val = input("Ingrese numero de clusters (>0): ")
    while not val.isdigit() or (val.isdigit() and int(val) <= 0):
        val = input("Ingrese numero de clusters (>0): ")
    return int(val)


def K_means(X:np.array,nunmClusters,maxIter=10):
    #Selecciono nunmClusters elementos aleatorios de X como centroides iniciales
    indices = [elem for elem in range(len(X))]   
    c = X[np.random.choice(indices , size=nunmClusters, replace=False, p=None)]
    c = c[:None]
    #Itero las veces definidas
    for i in range(maxIter):
        distancias = distanciaCentroides(X,c) 
        indiceClusterElem = indicesMinDist(distancias)
        for i in range(len(c)):
            c[i] = np.mean(X[indiceClusterElem==i],axis=0)
    
    #Calculos distancias e indices finales
    distancias = distanciaCentroides(X,c) 
    indiceClusterElem = indicesMinDist(distancias)    
    return (c,indiceClusterElem)
    

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    # Usuario selecciona cantidad cluster
    nunClusters = seleccionarClusters()
    #defino dataset de prueba
    dataset = np.array([[ 1.,3.,3.], [ 4.,5.,6.] , [7.,8.,9.] ])
    #calculo los centroides y clusters, con 8 iteraciones
    centroides,clusters = K_means(dataset,nunClusters,8)
    print("Dataset inicial:\n ",dataset)
    print("Centroides calculados:\n ",centroides)
    print("Cluster al que pertenece cada fila del dataset:\n ",clusters)