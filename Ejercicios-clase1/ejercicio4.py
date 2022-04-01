import numpy as np

def calculaMetricas(truth:np.array,prediction:np.array):
    '''Calcula a partir de 2 vectores de verdad y prediccion las metricas de Precision, Recall y Accuracy
       Se obtiene como salida 3 vectores como tupla, con los valores de Precision, Recall y Accuracy   
    '''
    TP = ((truth == 1 ) & (prediction == 1)).sum()
    TN = ((truth == 0 ) & (prediction == 0)).sum()
    FN = ((truth == 1 ) & (prediction == 0)).sum()
    FP = ((truth == 0 ) & (prediction == 1)).sum()

    Precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    Accuracy = (TP+TN) / (TP+TN+FN+FP)
    return Precision,Recall,Accuracy


#################################
# Prubeas unitarias del ejercicio

if __name__ == '__main__':
    truth = np.array([1,1,0,1,1,1,0,0,0,1])
    prediction = np.array([1,1,1,1,0,0,1,1,0,0])
    print("Vector de verdad: ",truth)
    print("Vector de prediccion: ",truth)

    Precision,Recall,Accuracy = calculaMetricas(truth,prediction)
    print("\nValores caldulados: ")
    print("Precision: ",Precision)
    print("Recall: ",Recall)
    print("Accuracy: ",Accuracy)