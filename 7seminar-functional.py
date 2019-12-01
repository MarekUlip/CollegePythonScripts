from sklearn import svm
import numpy as np
import random

#Natrenovani klasifikatoru s vice parametry napr jine c a g a porovnani presnosti
#nejcasteji se meni typ kernelu a parametr c 

params_list = [
    ['sep.csv',2,80,2,True,[]], 
    ['nonsep-orig.csv',2,112,2,True,[]],
    ['iris.csv',4,120,3,True,[]]]

kernels = ['linear','rbf','poly']
svm_params = [1,10,100,1000,10000]

def count_accuracy(classes,predicts):
    hits = 0
    if len(classes) != len(predicts):
        print('weird')
    for index, class_num in enumerate(classes):
        if class_num == predicts[index]:
            hits += 1
    print('Accuracy is {}'.format(hits/len(predicts)))
    return hits/len(predicts)
results = []
for params in params_list:
    print('\n\ntesting dataset {}'.format(params[0]))
    arr = np.loadtxt(params[0],delimiter=';',dtype=str)
    target_index = params[1]
    test_size = int(len(arr)*0.8)
    split_index = params[2]
    class_count = params[3]
    arr = arr.tolist()
    random.shuffle(arr)
    arr = np.array(arr)
    classes = arr[:,target_index].astype(int)
    classes = np.reshape(classes,(len(classes)))
    arr = arr[:,:target_index].tolist()

    for svm_param in svm_params:
        print()
        for kernel in kernels:
            print('Testing with params {}'.format([kernel,svm_param]))
            clf = svm.SVC(C=svm_param,gamma='scaled',kernel=kernel)
            clf.fit(arr[:test_size],classes[:test_size])
            predicts = clf.predict(arr[test_size:])
            results.append([count_accuracy(classes[test_size:],predicts),[kernel,svm_param],params[0]])

[print(item) for item in sorted(results,key=lambda x: x[0])]
        
            