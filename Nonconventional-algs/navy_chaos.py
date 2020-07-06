import math
import numpy as np


def sigm_(x):
    return x * (1 - x)


def sigm(x):
    return 1 / (1 + np.exp(-x))


def count_error(expected, predicted):
    if type(expected) is list:
        errors = [item - predicted for item in expected]
        minimum = 999999
        minimum_index = -1
        for index, error in enumerate(errors):
            if math.fabs(error) < minimum:
                minimum_index = index
                minimum = math.fabs(error)
        return errors[minimum_index]
    else:
        return expected - predicted


def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] + bias for item in input])


def predict(point, bias, weightsH, weightsO):
    inpts = [[[point[0], weightsH[0]], [point[1], weightsH[1]]], [[point[0], weightsH[2]], [point[1], weightsH[3]]]]
    outsH = []
    outsH.append(sigm(dot_product(inpts[0], bias[0])))
    outsH.append(sigm(dot_product(inpts[1], bias[1])))
    inpts = [[outsH[0], weightsO[0]], [outsH[1], weightsO[1]]]
    out = sigm(dot_product(inpts, bias[2]))
    return outsH, out


"""def backpropagationn(error, outO, outsH, weightsO, weightsH, learnConst, inputs):
    dO = error * sigm_(outO)
    dH = [dO*weightO*sigm_(outH) for weightO, outH in zip(weightsO, outsH)]
    inputs = [inputs[0], inputs[1], inputs[0], inputs[1]]
    #bias = [1,1]
    weightsO = [weightO+learnConst*dO*outH for weightO, outH in zip(weightsO, outsH)]
    for i in range(len(weightsH)):
        weightsH[i] = weightsH[i]+learnConst*dH[i%2]*inputs[i]
    #weightsH = [weightH*learnConst*dH*inpt for weightH, inpt in zip(weightsH,inputs)]
    #weightsH = [weightH*learnConst*dH*inpt for weightH, inpt in zip(weightsH,inputs)]
    return weightsO, weightsH"""


def backpropagation(error, outO, outsH, weightsO, weightsH, learnConst, inputs, biases):
    dO = error * sigm_(outO)
    dH = [dO * weightO * sigm_(outH) for weightO, outH in zip(weightsO, outsH)]
    weightsO = [weightO + learnConst * dO * outH for weightO, outH in zip(weightsO, outsH)]
    for i in [0, 1]:
        weightsH[i] += learnConst * dH[0] * inputs[i % 2]
    for i in [2, 3]:
        weightsH[i] += learnConst * dH[1] * inputs[i % 2]

    """biases[0] = adapt_bias(biases[0],error,learnConst)
    biases[1] = adapt_bias(biases[1],error,learnConst)"""

    biases[0] = adapt_bias(biases[0], dH[0], learnConst)
    biases[1] = adapt_bias(biases[1], dH[1], learnConst)
    biases[2] = adapt_bias(biases[2], dO, learnConst)

    return weightsO, weightsH, biases


def adapt_bias(bias, error, leaning_const):
    return bias + error * leaning_const


def start_prog(epochs, a, forced=True, use_chaos=False):
    # 2.5 3.3 4.0
    # a = 2.5
    params = None
    if a == 2.5:
        if forced:
            params = 0.6
        """if use_chaos:
            train_set, results = create_train_set_chaos(a, 1000, params)
        
        train_set, results = create_train_set(a, 1000, params)"""
    elif a == 3.3:
        if forced:
            params = [0.4794270198242338, 0.823603283206069]
        # train_set, results = create_train_set(a, 1000, params)  # 0.6)
    else:
        params = None
        # train_set, results = create_train_set(a, 1000, params)  # 0.6)

    if use_chaos:
        train_set, results = create_train_set_chaos(a, 1000, params)
    else:
        train_set, results = create_train_set(a, 1000, params)
    ##
    # inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # xored = [0, 1, 1, 0]
    learn_const = 0.2
    weightsH = [np.random.uniform() for i in range(4)]
    weightsO = [np.random.uniform() for i in range(2)]
    biases = [1, 1, 1]
    for epoch in range(epochs):
        error_sum = 0
        for index, point in enumerate(train_set):
            outsH, out = predict(point, biases, weightsH, weightsO)
            error = count_error(results[index], out)
            error_sum += math.fabs(error)
            weightsO, weightsH, biases = backpropagation(error, out, outsH, weightsO, weightsH, learn_const, point,
                                                         biases)
        # if error_sum < 0.4:
        # print("{}: {}".format(epoch,error_sum))
    print(weightsH)
    print(weightsO)
    print(biases)
    if use_chaos:
        train_set, results = create_test_set_chaos(a, 1000)
    else:
        train_set, results = create_test_set(a, 1000)

    max_diff = 0
    avg_difference = 0
    for index, point in enumerate(train_set):
        # print(predict(point, biases, weightsH, weightsO))
        predicted = predict(point, biases, weightsH, weightsO)[1]
        difference = results[index] - predicted
        avg_difference += math.fabs(difference)
        if math.fabs(difference) > max_diff:
            max_diff = math.fabs(difference)
        print("Point {} is expeted to be {} and was predicted as {}. Difference is {}".format(point, results[index],
                                                                                              predicted, difference))
    print("Max difference was {}. Average difference was: {}".format(max_diff, avg_difference / len(train_set)))


def create_train_set(a, count, result=None):
    base_x = np.random.uniform()
    train_set = []
    results = []
    for i in range(count):
        train_set.append([base_x, a])
        if result is None:
            results.append(a * base_x * (1 - base_x))
        else:
            if type(result) is list:
                expected_result = a * base_x * (1 - base_x)
                errors = [item - expected_result for item in result]
                minimum = 999999
                minimum_index = -1
                for index, error in enumerate(errors):
                    if math.fabs(error) < minimum:
                        minimum_index = index
                        minimum = math.fabs(error)
                # print(result[minimum_index])
                results.append(result[minimum_index])
            else:
                results.append(result)
        base_x = np.random.uniform()
    return train_set, results


def create_test_set(a, count):
    base_x = np.random.uniform()
    train_set = []
    results = []
    for i in range(count):
        train_set.append([base_x, a])
        results.append(a * base_x * (1 - base_x))
        base_x = np.random.uniform()
    return train_set, results


def create_train_set_chaos(a, count, result=None):
    base_x = 0.01
    train_set = []
    results = []
    for i in range(count):
        train_set.append([base_x, a])
        if result is None:
            results.append(a * base_x * (1 - base_x))
        else:
            if type(result) is list:
                expected_result = a * base_x * (1 - base_x)
                errors = [item - expected_result for item in result]
                minimum = 999999
                minimum_index = -1
                for index, error in enumerate(errors):
                    if math.fabs(error) < minimum:
                        minimum_index = index
                        minimum = math.fabs(error)
                # print(result[minimum_index])
                results.append(result[minimum_index])
            else:
                results.append(result)
        base_x = a * base_x * (1 - base_x)
    return train_set, results


def create_test_set_chaos(a, count):
    base_x = np.random.uniform()
    train_set = []
    results = []
    for i in range(count):
        train_set.append([base_x, a])
        results.append(a * base_x * (1 - base_x))
        base_x = a * base_x * (1 - base_x)
    return train_set, results


start_prog(500, 4.0, False, False)
