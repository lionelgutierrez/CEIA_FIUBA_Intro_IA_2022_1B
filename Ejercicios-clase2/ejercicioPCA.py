import numpy as np
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

def PCA_ej2(X:np.array,num_dimensiones:int):
    '''Dado X de dimensiones n,d, con n muestras y d features
       El objetivo es reducir sus dimensiones a num_dimensiones 
       Parametros entrada: 
            X: array con datos
            num_dimensiones: cantidad componentes a utilizar
       Salidas: tupla de array autovalores mas significativos, array de PCA con num_dimensiones

    '''
    #Paso 1: centrar el dataset (Hint: usen np.mean)
    X = X - np.mean(X,axis=0)
    #Paso 2 obtener matriz de covarianza a partir de X transpuesta
    matCov = np.cov(X.T)
    #Paso 3 cualculo autovectores y autovalores de matCov
    autval,autvec = np.linalg.eig(matCov)
    #Paso 4 ordeno los autovectores en sentido de autovalores decrecientes
    indicesAuvOrden = np.argsort(autval*-1)
    autval = autval[indicesAuvOrden]
    autvec = autvec[:,indicesAuvOrden]
    # Proyecto el dataset sobre los autovectores mas relevantes (cantidad segun num_dimensiones)
    # Retorno los autovalres mas relevantes y la proyeccion
    return autval[:num_dimensiones], X.dot(autvec[:,:num_dimensiones])

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    #Defino cantidad componentes a tomar
    num_componentes = 1
    # Defino array de prueba
    X = np.array([[0.8, 0.7], [0.1, -0.1]])
    #Calculo PCA con el algoritmo desarrollado
    pca_calc_ej2 = PCA_ej2(X,num_componentes)

    #Calculo PCA con la libreria de Sklearn
    pca = PCA(n_components=num_componentes)
    pca.fit(X)
    pca_cal_sklearn = pca.transform(X)

    print("Matriz sobre la que calculo PCA: \n",X)    
    print("Resultado PCA calculado en numpy: ",pca_calc_ej2[1])
    print("Resultado PCA con Scikit-learn: ",pca_cal_sklearn)
    np.testing.assert_allclose(pca_cal_sklearn, pca_calc_ej2[1])
    print("Autovalores principales: ",pca_calc_ej2[0])

