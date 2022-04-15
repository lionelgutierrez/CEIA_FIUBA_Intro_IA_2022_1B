import numpy as np

class MetricaBase():
    '''Clase base para las metricas. La misma guarda los parametros necesarios para cada metrica a calcular'''
    def __init__(self,**kwargs):
        self._parametros = kwargs

    def __call__(self, *args, **kwargs):
        pass        

class MSE(MetricaBase):
    '''Clase para el calculo de la metrica MSE'''
    def __init___(self,**kwargs):
        MetricaBase.__init__(self,**kwargs)

    def __call__(self):
        if "truth" in self._parametros and "prediction" in self._parametros:
            truth = self._parametros["truth"]
            n = truth.size
            prediction = self._parametros["prediction"]
            return (np.sum( (truth-prediction) ** 2) / n)
        return None    
