import numpy as np

class Indexador():
    '''Esta clase genera un indexador para identificadores de usaurio'''
    def __init__(self,ids:np.array):
        id2idx = np.ones(np.amax(ids)+1, dtype=int) * -1
        elems,indices = np.unique(ids,return_index=True)
        id2idx[elems]=indices
        idx2id = np.zeros((len(indices),),dtype=int)
        idx2id[indices]=elems
        self.__id2idx = id2idx
        self.__idx2id = idx2id
        
    def get_users_id(self,idxs:np.array):
        '''Obtiene los ids de usaurio a partir de un vector de indices'''
        return self.__idx2id[idxs]
    
    def get_users_idx(self,ids:np.array):
        '''Obtiene los indices de usaurio a partir de un vector de ids de usaurio'''
        ids_filtrados = self.__id2idx[ids]
        return ids_filtrados
    
#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    user_ids = np.array([15,12,14,10,1,2,1], dtype=int)
    print("Identificadores de usuario: ",user_ids)
    indexador = Indexador(user_ids)
    print("Consulto user_idx para el vector original y obtengo: ", indexador.get_users_idx(user_ids))
    vectorConsulta1 = [15,0,3,12]
    print("Consulto user_idx para el vector ",vectorConsulta1, " y obtengo: " ,indexador.get_users_idx(vectorConsulta1))
    vectorConsulta2 = [0,5,3,4]
    print("Consulto user_id para el vector de indices ",vectorConsulta2, " y obtengo: ",indexador.get_users_id(vectorConsulta2))