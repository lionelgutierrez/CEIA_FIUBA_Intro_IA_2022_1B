import numpy as np

def distanciaCentroides(X:np.array,C:np.array):
    '''Dada una nube de puntos X y centroides C, obtiene la distancia entre cada vector X y los centroides C'''
    return np.linalg.norm(X-C[:,np.newaxis,:],axis=2)

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    X = np.array([[ 1.,2.,3.], [ 4.,5.,6.] , [7.,8.,9.] ])
    c = np.array([[ 1.,0.,0.] , [0.,1.,1.] ])
    print("Matriz de puntos: \n",X)
    print("Centorides: \n",c)
    distancia = distanciaCentroides(X,c) 
    print("Distancia calculada: \n",distancia)