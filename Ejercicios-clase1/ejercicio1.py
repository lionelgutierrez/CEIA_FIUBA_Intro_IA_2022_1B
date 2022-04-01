import numpy as np

def norma_l0(matriz:np.array):
    '''Calcula la norma l0  para un array de numpy'''
    return np.count_nonzero(matriz,axis=1)

def norma_l1(matriz:np.array):
    '''Calcula la norma l1  para un array de numpy'''
    return np.sum(np.absolute(matriz),axis=1)

def norma_l2(matriz:np.array):
    '''Calcula la norma l2  para un array de numpy'''
    return np.sqrt(np.sum(matriz ** 2, axis=1))

def norma_li(i:int,matriz:np.array):
    '''Calcula la norma li para un i positivo generico'''
    if i>1:
        return np.round((np.sum(np.absolute(matriz)**i,axis=1))**(1/i),decimals=2)
    else:
        return None    

def norma_inf(matriz:np.array):
    '''Calcula la norma infinita para un array de numpy'''
    return np.argmax(np.absolute(matriz),axis=1)


#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    a1 = np.arange(16).reshape(4,4)
    a1=a1*-1
    a2 = np.arange(16).reshape(2,8)
    a3 = np.arange(16).reshape(1,16)

    print("matriz a1: ")
    print(a1)
    print("norma l0: ",norma_l0(a1))
    print("norma l1: ",norma_l1(a1))
    print("norma l2: ",norma_l2(a1))
    print("norma inf: ",norma_inf(a1))
    print("\nmatriz a2: ")
    print(a2)
    print("norma l0: ",norma_l0(a2))
    print("norma l1: ",norma_l1(a2))
    print("norma l2: ",norma_l2(a2))
    print("norma inf: ",norma_inf(a2))
    print("\nmatriz a3: ")
    print(a3)
    print("norma l0: ",norma_l0(a3))
    print("norma l1: ",norma_l1(a3))
    print("norma l2: ",norma_l2(a3))
    print("norma inf: ",norma_inf(a3))