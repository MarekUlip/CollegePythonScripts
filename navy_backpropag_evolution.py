import math
import numpy as np


def sigm(x):
    return 1/(1+np.exp(-x))

def count_error(expected, predicted):
    return expected - predicted

def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] for item in input])

def predict(point, weightsH, weightsO, bias=(1,1)):
    inpts = [[[point[0],weightsH[0]], [point[1],weightsH[1]]], [[point[0],weightsH[2]], [point[1],weightsH[3]]]]
    outsH = []
    outsH.append(sigm(dot_product(inpts[0], bias[0])))
    outsH.append(sigm(dot_product(inpts[1],bias[0])))
    inpts = [[outsH[0],weightsO[0]], [outsH[1],weightsO[1]]]
    out = sigm(dot_product(inpts,bias[1]))
    return outsH, out

def normal_distribution(mean, scale, min_max=None):
    element = np.random.normal(mean, scale)
    if min_max is not None:
        wrong = True
        while wrong:
            wrong = False
            if (element < min_max[0]) or (element > min_max[1]):
                wrong = True
                element = np.random.normal(mean, scale)

    return element

def generate_identity(dim):
    res = []
    for i in range(dim):
        res.append(np.random.uniform())
    return res

def generate_identity_with_normal(prev, scale, min_max=None):
    res = []
    for x in prev:
        res.append(normal_distribution(x, scale,min_max))
    if len(res) == 0:
        print(res)
    return res

def hill_climbing(minimum, maximum, dim, cycles, num_of_identities, scale, test_func):
    all_res = []
    best = generate_identity(dim)
    fitness = test_func(best)
    best_tmp = []
    for i in range(cycles):
        for j in range(num_of_identities):
            tmp = generate_identity_with_normal(best, scale, [minimum, maximum])
            all_res.append(tmp)
            fit_tmp = test_func(tmp)
            print(fit_tmp)
            if fit_tmp < fitness:
                fitness = fit_tmp
                best_tmp = tmp
        if len(best_tmp) == len(best):
            best = best_tmp
    return best

def test_function(weights):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    error_sum = 0
    for index, point in enumerate(inputs):
        _, out = predict(point,weights[:4], weights[4:])
        error = count_error(xored[index], out)
        error_sum += math.fabs(error)
    return error_sum

def start_prog():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    weights = hill_climbing(-30.0,30.0,6,500,100,10, test_function)


    for index, point in enumerate(inputs):
        #print(predict(point, biases, weightsH, weightsO))
        print("Point {} is expeted to be {} and was predicted as {}".format(point, xored[index], predict(point,weights[:4], weights[4:])[1]))

start_prog()