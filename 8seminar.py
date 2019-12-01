from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import random
import numpy as np



random.seed(4849)
params_list = [
    ['sep.csv',2,80,2,True,[]], 
    ['nonsep-orig.csv',2,112,2,True,[]],
    ['iris.csv',4,120,3,True,[]]]

def create_variations(depth, field, all_vars, possibilities):
    if depth == len(all_vars):
        possibilities.append(field)
        return

    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth + 1, f, all_vars, possibilities)

def count_accuracy(classes,predicts):
    hits = 0
    if len(classes) != len(predicts):
        print('weird')
    for index, class_num in enumerate(classes):
        if class_num == predicts[index]:
            hits += 1
    #print('Accuracy is {}'.format(hits/len(predicts)))
    return hits/len(predicts)

def create_layers(num_of_layer, layer_size):
    layers = tuple([layer_size for i in range(num_of_layer)])
    return layers
num_of_layers = [1,4]
num_of_neurons = [i for i in range(10,300,10)]
activation_functions = ['relu','tanh','identity','logistic']
solvers = ['lbfgs','adam','sgd']
all_vals = [num_of_layers,num_of_neurons,activation_functions,solvers]
variations = []
create_variations(0,[],all_vals,variations)
best_accuracy = 0
best_params = []
print(len(variations))

dataset_best_params = []
testable_indexes = [2]

for index, params in enumerate(params_list):
    if index not in testable_indexes:
        continue
    print('\n\ntesting dataset {}'.format(params[0]))
    arr = np.loadtxt(params[0],delimiter=';',dtype=str)
    target_index = params[1]
    test_size = int(len(arr)*0.8)
    split_index = params[2]
    class_count = params[3]
    arr = arr.tolist()
    random.shuffle(arr)
    arr = np.array(arr)
    scaler = StandardScaler()
    arr = scaler.fit_transform(arr)
    classes = arr[:,target_index].astype(int)
    classes = np.reshape(classes,(len(classes)))
    arr = arr[:,:target_index].astype(float).tolist()
    for index,variation in enumerate([1]):#enumerate(variations):
        classifier = MLPClassifier(hidden_layer_sizes=[20],activation='tanh',solver='adam')#MLPClassifier(hidden_layer_sizes=create_layers(variation[0],variation[1]),activation=variation[2],solver=variation[3],max_iter=300)
        #classifier = MLPClassifier(hidden_layer_sizes=[40,40,40,40],activation='tanh',solver='lbfgs') #nonsep
        classifier.fit(arr[:test_size],classes[:test_size])
        predicts = classifier.predict(arr[test_size:])
        accuracy = count_accuracy(classes[test_size:],predicts)
        if index%10 == 0:
            print(index)
        if accuracy >= best_accuracy:
            if accuracy > best_accuracy:
                best_params.clear()
                print(accuracy)
            best_accuracy = accuracy
            best_params.append(variation)
            #
            #print(variation)
    dataset_best_params.append(best_params[:])
    best_params.clear()
    best_accuracy = 0
            
for best_d_params in dataset_best_params:
    print(len(best_d_params))
    [print(params) for params in best_d_params]

