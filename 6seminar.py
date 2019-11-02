from sklearn import preprocessing
import numpy as np

def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    return x_scaled


def create_decision_tree(datas):
    num_of_features = len(datas)
    for i in num_of_features:
        for j in range(0,1,0.1):
            

def gini_index(groups, dataset_size):
    sum_all = 0
    for group in groups:
        proportion = len(group)/dataset_size
        sum_all+=proportion**2
    return 1-sum_all

def split_datas(index, value, data):
    lesser = []
    greater = []
    for row in data:
        if row[index]<value:
            lesser.append(row)
        else:
            greater.append(row)
    return lesser, greater

#Rozdelit si sadu a ukazat jaka je presnost pomoci confusion matrix

arr = np.loadtxt('iris.csv',delimiter=';')
print(normalize(arr[:,:4]))
