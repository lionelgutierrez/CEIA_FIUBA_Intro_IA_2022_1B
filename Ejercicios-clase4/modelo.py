import numpy as np

class ModeloBase(object):
    def __init__(self):
        self._modelo = None

    def fit(self,X,Y):
        return NotImplemented

    def predict(self,X):
        return NotImplemented

class RegresionLineal(ModeloBase):
    def fit(self,X,Y):
        Beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        self._modelo = Beta
        
    def predict(self,X):
        y_hat = X @ self._modelo 
        return y_hat
    
    
class RegLineConTermInd(ModeloBase):
    def fit(self,X,Y):
        X_expandida = np.vstack((X.T, np.ones(len(X)))).T
        Beta = np.linalg.inv(X_expandida.T.dot(X_expandida)).dot(X_expandida.T).dot(Y)        
        self._modelo = Beta
        
    def predict(self,X):
        X_expandida = np.vstack((X.T, np.ones(len(X)))).T
        y_hat = X_expandida @ self._modelo 
        return y_hat    

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    X = np.array([[1,1,2,2],[2,2,2,3],[3,4,3,3],[4,4,4,4],[5,5,5,5]])
    Y = np.array([1,2,3,4,5])
    print("Valores de X: \n",X)
    print("Valores de Y: \n",Y)
    r = RegresionLineal()
    r.fit(X,Y)
    print("Prediccion de 2*X con el modelo simple: ",r.predict(X*2))
    Xi= np.array([[2,2,1,1],[3,3,3,4],[4,5,4,4],[5,5,5,5],[6,6,6,6]])
    Yi = np.array([2,3,4,5,6])
    ri = RegLineConTermInd()
    ri.fit(X,Yi)
    print("Modelo: ",ri._modelo)
    print("Prediccion de 2*Xi con el modelo con termino independiente: ",ri.predict(Xi*2))
    