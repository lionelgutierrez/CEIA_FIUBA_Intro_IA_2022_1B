from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal

class BaseModel(object):
    def __init__(self):
        self.model = None
    def fit(self, X, Y):
        return NotImplemented
    def predict(self, X):
        return NotImplemented

class EMScalar(BaseModel):

    def fit(self, X ,k, max_iteraciones=500,tol_corte = 1E-5 ):    
        # Dimensions
        n = X.shape[0]

        # Inicialización de parámetros
        # Inicializar las probabilidades marginales de las clases P(z)
        # Inicializo como una uniforme para cada clase
        p = np.ones(k) / k
        p = p.reshape(-1,1)
        
        # Inicializar medias
        #means = np.random.uniform(np.min(X), np.max(X), (k, 1)) # k medias en intervalo (min(x),max(x))
        #Tomo como medias, los centroides calculados con Kmeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        means = kmeans.cluster_centers_
                
        # Inicializar matrices covarianza
        covariance = np.sum((np.hstack((X, X))-means.T)**2, axis=0)/(X.shape[0]-1)
        covariance = covariance.reshape(-1, 1)
           
        # Crear matrices place-holders para 
        # p(x|z) para cada clase z (n x k), [p(x1|z1) p(x1|z2) p(x1|z3) ..]
        # Responsibilities
        Nij = np.zeros((n, k))
        Eij = np.zeros((n, k))
        Eij_ant = np.zeros((n, k))

        i = 0
        delta = False
        # Calcular, con los parámetros iniciales, p(x|z) para todos los z
        for j in range(k):
            Nij[:, j] = multivariate_normal.pdf(X, means[j], covariance[j])
    
        while not (delta or i > max_iteraciones):
            Eij_ant[:, :] = Eij
            for j in range(k):
                # Responsibilities
                Eij[:, j] = (p[j] * Nij[:, j]) / (Nij @ p)[:, 0]
                # Nij shape:  (400, 2)
                #p shape:  (2, 1)
                # (Nij @ p)[:, 0] shape:  (400, 1)
                
                # Actualizar medias 
                means[j] = (Eij[:, j].dot(X)) / np.sum(Eij[:, j], axis=0)
                # Actualizar covarianzas
                covariance[j] = Eij[:, j].dot((X - means[j]) * (X - means[j])) / np.sum(Eij[:, j])
                # Actualizar pesos de clases
                p[j] = np.mean(Eij[:, j])
                # Actualizar p(x|z)
                Nij[:, j] = multivariate_normal.pdf(X, means[j], covariance[j])
            delta = np.allclose(Eij_ant, Eij, rtol=tol_corte)
            i = i + 1
        idx = np.argsort(means[:, 0], axis=0)

        # Al finalizar el loop, guardar el modelo en la clase
        self.model = {'mu': means[idx, :], 'cov': covariance[idx, :], 'p': p[idx, :]}        
        

    def predict(self, X):
        
        k = self.model['mu'].shape[0]
        N = np.zeros((X.shape[0], k))
        E = np.zeros((X.shape[0], k))

        for i in range(k):
            N[:, i] = multivariate_normal.pdf(X, self.model['mu'][i, 0], self.model['cov'][i, 0])
        for i in range(k):
            E[:, i] = (self.model['p'][i, 0] * N[:, i]) / (N @ self.model['p'])[:, 0]
        idx = np.argmax(E, axis=1)
        return idx