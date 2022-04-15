import numpy as np
import csv



class Data(object):
    '''Clase que carga un dataset y hace el split del mismo, segun archivo cvs de 7 columnas
       donde los features son las primeras 6 columnas y la septima es el valor de Y
    '''
    def __init__(self,path:str):
        self._dataset = self.cargarDataset(path)

    @staticmethod
    def cargarDataset(path:str):
        with open(path,'r') as dest_f:
            data_iter = csv.reader(dest_f,
                                delimiter = ";")
            data = [data for data in data_iter]
        data_array = np.asarray(data, dtype = float)
        return data_array

    def dividir(self,porcentaje:float):
        '''Dado un dataset, permuta el mismo y lo divie en 2 dataset nuevos con el porcentaje , 1- porcentaje de los datos
        Devuelve:
                trainX: Los valores de X de train con cantidad (porcentaje) % del total de datos
                trainY: Los valores de Y de train con cantidad (porcentaje) % del total de datos
                testX: Los valores de X de test con cantidad (1-porcentaje) % del total de datos
                testY: Los valores de Y de test con cantidad (1-porcentaje) % del total de datos
        '''           
        if porcentaje > 1.0 or porcentaje <0:
            porcentaje_split = 0.8
        else:
            porcentaje_split = porcentaje
        
        permutado = np.random.permutation(self._dataset)
        train_set,test_set = np.split(permutado,[int(porcentaje_split*len(self._dataset))])

        return train_set[:,:6],train_set[:,6],test_set[:,:6],test_set[:,6]

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    midata = Data("clase3v2.csv")    
    print("Shape del dataset: ",midata._dataset.shape)
    testX, testY, trainX,trainY = midata.dividir(0.8)
    print("Shape de la X del dataset de test: ",testX.shape)
    print("Shape de la Y del dataset de test: ",testY.shape)
    print("Shape de la X dataset de train: ",trainX.shape)
    print("Shape de la Y  dataset de train: ",trainY.shape)