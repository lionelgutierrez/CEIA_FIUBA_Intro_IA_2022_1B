import numpy as np

class MetricaBase():
    '''Clase base para las metricas. La misma guarda los parametros necesarios para cada metrica a calcular'''
    def __init__(self,**kwargs):
        self._parametros = kwargs

    def __call__(self, *args, **kwargs):
        pass        

class Precision(MetricaBase):
    '''Clase para el calculo de la metrica Precision'''
    def __init___(self,**kwargs):
        MetricaBase.__init__(self,**kwargs)
    
    def __call__(self):
        if "truth" in self._parametros and "prediction" in self._parametros:
            truth = self._parametros["truth"]
            prediction = self._parametros["prediction"]
            TP = ((truth == 1 ) & (prediction == 1)).sum()
            FP = ((truth == 0 ) & (prediction == 1)).sum()
            Precision = TP / (TP+FP)
            return (Precision)
        return None    

class Recall(MetricaBase):
    '''Clase para el calculo de la metrica Recall'''
    def __init___(self,**kwargs):
        MetricaBase.__init__(self,**kwargs)
    
    def __call__(self):
        if "truth" in self._parametros and "prediction" in self._parametros:
            truth = self._parametros["truth"]
            prediction = self._parametros["prediction"]
            TP = ((truth == 1 ) & (prediction == 1)).sum()
            FN = ((truth == 1 ) & (prediction == 0)).sum()
            Recall = TP / (TP+FN)
            return (Recall)
        return None   

class Accuracy(MetricaBase):
    '''Clase para el calculo de la metrica Accuracy'''
    def __init___(self,**kwargs):
        MetricaBase.__init__(self,**kwargs)
    
    def __call__(self):
        if "truth" in self._parametros and "prediction" in self._parametros:
            truth = self._parametros["truth"]
            prediction = self._parametros["prediction"]
            TP = ((truth == 1 ) & (prediction == 1)).sum()
            TN = ((truth == 0 ) & (prediction == 0)).sum()
            FN = ((truth == 1 ) & (prediction == 0)).sum()
            FP = ((truth == 0 ) & (prediction == 1)).sum()
            Accuracy = (TP+TN) / (TP+TN+FN+FP)
            return (Accuracy)
        return None   

class Avg_precision(MetricaBase):
    '''Clase para el calculo de la metrica Avg_Precision'''
    def __init___(self,**kwargs):
        MetricaBase.__init__(self,**kwargs)
    
    def __call__(self):
        if "q_id" in self._parametros and "truth_relevance" in self._parametros:
            q_id = self._parametros["q_id"]
            truth_relevance = self._parametros["truth_relevance"]
            ids_unicos = np.unique(q_id)
            masc_positivos = (truth_relevance == 1)
            q_id_positivos = q_id[masc_positivos]
            cant_positivos_por_id = np.bincount(q_id_positivos)
            cant_total_por_id = np.bincount(q_id)
            cant_positivos_por_id = cant_positivos_por_id[ids_unicos]
            cant_total_por_id = cant_total_por_id[ids_unicos]
            precision_por_q_id = cant_positivos_por_id / cant_total_por_id
            metrica = np.sum(precision_por_q_id) / len(ids_unicos)
            return metrica
        return None




class IteraMetricas:
    '''Clase que permite iterar entre diferentes metricas definidas
       Recibe un parametro con los vectores necesarios para las diferentes metricas 
        '''
    def __init__(self,**kwargs):
        self.__parametros = kwargs
        self.__metricasCalculadas = {}
        self.__listaMetricas = [Avg_precision,Accuracy,Recall,Precision]

    def obtener_metricas(self):
        '''Metodo que permite calcular las metricas definidas en la clase
           Devuelve un diccionario con el nombre de la metrica y el valor obtenido para la misma en funcion de los
           vectores indicados en la creacion del objeto iteraMetricas     
        '''
        for metrica in self.__listaMetricas:
            m = metrica(**self.__parametros)
            self.__metricasCalculadas[metrica.__name__] = m()
        return  self.__metricasCalculadas   

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    #Defino los vectores a utilizar para el calculo de las metricas
    truth = np.array([1,1,0,1,1,1,0,0,0,1])
    prediction = np.array([1,1,1,1,0,0,1,1,0,0])

    q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    truth_relevance = np.array([True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True ])

    #Creo el objeto iterador de metricas, con los vectores de datos definidos
    iterador = IteraMetricas(truth=truth,prediction=prediction,q_id=q_id,truth_relevance=truth_relevance)
    #Muestro resultados obtenidos
    print("Vectores utilizados para los calculos\n" )
    print("vector truth: ",truth)
    print("vector prediction: ",prediction)
    print("vector q_id: ",q_id)
    print("vector truth_relevance: ",truth_relevance)
    print("\nMetricas calculadas: \n" )
    diccionario_metricas = iterador.obtener_metricas()
    for metrica, valor in diccionario_metricas.items():
        print(metrica, " : ", valor)

