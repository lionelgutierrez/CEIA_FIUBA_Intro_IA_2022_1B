import numpy as np

def separaDataset(dataset):
    '''Dado un dataset, permuta el mismo y lo divie en 3 dataset nuevos con el 70%, 20% y 10% de los datos
       Devuelve:
            train_set: 70% de los datos
            valid_set: 20% de los datos
            test_set: 10% de los datos
    '''        
    
    train_set,valid_set,test_set = np.split(np.random.permutation(dataset),[int(0.7*len(dataset)),int(0.9*len(dataset))])
    return train_set,valid_set,test_set

#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    n = 30
    m = 5
    dataset = np.arange(n*m,dtype="float").reshape(n,m)    
    print("Dataset original:\n",dataset)
    train_set,valid_set,test_set  = separaDataset(dataset) 
    print("\n Dataset de train:\n",train_set)    
    print("\n Dataset de tesvalidaciont:\n",valid_set)    
    print("\n Dataset de test:\n",test_set)    
