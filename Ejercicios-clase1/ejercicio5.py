import numpy as np

def avg_query_precision(q_id:np.array,truth_relevance:np.array):
    '''Calcula la metrica average query precision.
       Recibe 2 vectores de ids de consulta y el vector de verdad
       Calcula la presicion para cada id y luego calcula el promedio de las presiciones
       Devuelve: la metrica calculadad
    '''
    #Obtengo los ids unicos que tengo en q_id
    ids_unicos = np.unique(q_id)
    # Obtengo solo los Ids con resutlados positivos
    masc_positivos = (truth_relevance == 1)
    q_id_positivos = q_id[masc_positivos]
    # Cuento cantidad positivos por cada id
    cant_positivos_por_id = np.bincount(q_id_positivos)
    # Cuento cantidad total por cada id
    cant_total_por_id = np.bincount(q_id)
    #filtro solo los indices que son parte de los q_id de mis 2 nuevos vectores
    #dado que bincount incluye valores que pueden no ser parte del q_id como el 0
    cant_positivos_por_id = cant_positivos_por_id[ids_unicos]
    cant_total_por_id = cant_total_por_id[ids_unicos]
    #Hago la division elemento a elemento con broadcasting
    precision_por_q_id = cant_positivos_por_id / cant_total_por_id
    metrica = np.sum(precision_por_q_id) / len(ids_unicos)
    return metrica
    
#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':

    q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    predicted_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    truth_relevance = np.array([True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True ])
    print("Vector de q_ids: ",q_id)
    print("Vector de verdad: ",truth_relevance)
    print("\n Metrica calculada: ",avg_query_precision(q_id,truth_relevance))
    