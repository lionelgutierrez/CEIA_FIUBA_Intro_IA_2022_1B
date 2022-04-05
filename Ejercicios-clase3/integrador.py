import numpy as np
from Data import Data
from ejercicio3  import reemplazaNanPorMedia as reemplazaNanPorMedia
from metricas import MSE
from modelo import RegresionLineal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 

def removerNans(X,Y):
    '''Dado un par X e Y , a X le quita todas las filas y columnas que tengan algun nan, y elimina las filas de Y eliminadas de X
       Devuelve el dataset sin las filas y columnas con nans
    '''    
    filtroColumnas = ~(np.isnan(X).any(axis=0))
    filtroFilas = ~(np.isnan(X).any(axis=1))
    X = X[filtroFilas]
    Y = Y[filtroFilas]
    X = X[:,filtroColumnas]
    return X,Y

# Paso 1 implementar clase Data -  Cargar los datos con objeto de clase Data (implementada por ustedes) . Clase en Data.py
# Paso 2 cargo los datos con objeto de clase Data
midata = Data("clase3v2.csv")    

# Paso 3 Hacer split de datos en 80/20
trainX,trainY,testX,testY = midata.dividir(0.8)

# Paso 4.1 Tratar nans con eliminacion. Elimino tanto para X como para Y las filas para conservar la forma del dataset 
X_test_sin_nans,testY_sin_nans = removerNans(testX,testY)
X_train_sin_nans,trainY_sin_nans = removerNans(trainX,trainY)

# Paso 4.2 Tratar nans con media
X_test_nans_media = reemplazaNanPorMedia(testX)
X_train_nans_media = reemplazaNanPorMedia(trainX)

num_dimensiones = 3

# Paso 5.1 Aplicar PCA para quedarse con 3 dimensiones a 4.1
# Estandarizamos los datos
RANDOM_STATE = 17
scaler = StandardScaler()

media_X_train_sin_nans = np.nanmean(X_train_sin_nans,axis=0,keepdims=True).reshape(X_train_sin_nans.shape[1],1)
X_train_sin_nans_scaled = scaler.fit_transform(X_train_sin_nans)

#Calculo PCA con 3 dimensiones
pca_sin_nans = PCA(n_components=num_dimensiones, random_state=RANDOM_STATE).fit(X_train_sin_nans_scaled)
X_train_sin_nans_reducido_pca = (X_train_sin_nans_scaled.dot( pca_sin_nans.components_.T))#+ (media_X_train_sin_nans[:num_dimensiones]).T


# Paso 5.2 Aplicar PCA para quedarse con 3 dimensiones a 4.2
media_X_train_nans_media= np.nanmean(X_train_nans_media,axis=0,keepdims=True).reshape(X_train_nans_media.shape[1],1)
X_train_nans_media_scaled = scaler.fit_transform(X_train_nans_media)

#Calculo PCA con 3 dimensiones
pca_nan_media = PCA(n_components=num_dimensiones, random_state=RANDOM_STATE).fit(X_train_nans_media_scaled)
X_train_nans_media_reducido_pca = (X_train_nans_media_scaled.dot( pca_nan_media.components_.T))#+ (media_X_train_nans_media[:num_dimensiones]).T

# Paso 6 Usar clas metrica, crear clase MSE que hereda de esta
# Creada en archivo metricas.py

# Crear una clase modelo base y clase regresión lineal que herede de ella.  (esto viene de ejercicios anteirores)
# Creada en archivo modelo.py

# Paso 7 Calcular regresion lineal sobre train de 5.1 y 5.2

reg_sin_nans = RegresionLineal()
reg_sin_nans.fit(X_train_sin_nans_reducido_pca,trainY_sin_nans)

reg_nans_media = RegresionLineal()
reg_nans_media.fit(X_train_nans_media_reducido_pca,trainY)

# Paso 8 calcular MSE sobre validacion contra modelo sin nans y con media por nans y comparar
# Escalo los datos y calculo los componentes de PCA para cada caso
X_test_sin_nans = scaler.fit_transform(X_test_sin_nans)
X_test_nans_media = scaler.fit_transform(X_test_nans_media)
media_X_test_sin_nans  = np.nanmean(X_test_sin_nans,axis=0,keepdims=True).reshape(X_test_sin_nans.shape[1],1)
media_X_test_nans_media  = np.nanmean(X_test_nans_media,axis=0,keepdims=True).reshape(X_test_nans_media.shape[1],1)

X_test_sin_nans_reducido_pca = (X_test_sin_nans.dot( pca_sin_nans.components_.T))#+ (media_X_test_sin_nans[:num_dimensiones]).T
X_test_nans_media_reducido_pca = (X_test_nans_media.dot( pca_nan_media.components_.T))#+ (media_X_test_nans_media[:num_dimensiones]).T

Y_reg_sin_nans = reg_sin_nans.predict(X_test_sin_nans_reducido_pca)
Y_reg_nans_media = reg_nans_media.predict(X_test_nans_media_reducido_pca)

# Calculo la metrica MSE para las 2 Y obtenidas contra mi Y de train
mse_sin_nans = MSE(truth=testY_sin_nans,prediction=Y_reg_sin_nans)
mse_nans_media = MSE(truth=testY,prediction=Y_reg_nans_media)

print("MSE para el dataset con tratamiento de eliminar nans: ",mse_sin_nans())
print("MSE para el dataset con media para nans: ",mse_nans_media())

#Si no escalo los datos de test antes de reducir los componentes obtengo:
#MSE para el dataset con tratamiento de eliminar nans:  32256.079276692035
#MSE para el dataset con media para nans:  23657.686089238992

# Si escalo los datos y calculo los componentes de PCA para cada caso
#MSE para el dataset con tratamiento de eliminar nans:  153.37323514713935
#MSE para el dataset con media para nans:  296.2013160636221

# El MSE si escalo y comparo los datos de test y train escalados me dan que el metodo de eliminar nans es mejor que el metodo de la media
# para este caso.
# Si no escalara los datos de test el resultado es el opuesto

#Si escalo los datos y los ajusto por su media el valor que obtengo es peor
#MSE para el dataset con tratamiento de eliminar nans:  728.7883248687206
#MSE para el dataset con media para nans:  531.474378518453

