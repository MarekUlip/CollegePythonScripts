import math
import numpy as np


def dot_product(input):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sigm(sum([item[0] * item[1] for item in input]))


def sigm(x):
    return 1 / (1 + math.e ** (-x))


def count_error(expected, predicted):
    return predicted - expected


def predict(point, weights_h, weights_o):
    outsH = [dot_product([[point[0], weights_h[0 + 2 * i]], [point[1], weights_h[1 + 2 * i]]]) for i in range(2)]
    out = dot_product([[outsH[0], weights_o[0]], [outsH[1], weights_o[1]]])
    return outsH, out


def backpropagation(error, outO, outH, weightsO, weightsH, input, learn_const):
    weightsO[0] -= learn_const * (outH[0] * error)
    weightsO[1] -= learn_const * (outH[1] * error)
    weightsH[0] -= learn_const * (input[0] * error * weightsO[0])
    weightsH[1] -= learn_const * (input[1] * error * weightsO[0])
    weightsH[2] -= learn_const * (input[0] * error * weightsO[1])
    weightsH[3] -= learn_const * (input[1] * error * weightsO[1])
    return weightsO, weightsH


def start_prog(epochs):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    learn_const = 0.1
    weightsH = [np.random.uniform() for i in range(4)]
    weightsO = [np.random.uniform() for i in range(2)]
    for epoch in range(epochs):
        for index, point in enumerate(inputs):
            outsH, out = predict(point, weightsH, weightsO)
            error = count_error(xored[index], out)
            backpropagation(error, out, outsH, weightsO, weightsH, point, learn_const)

    for index, point in enumerate(inputs):
        # print(predict(point, biases, weightsH, weightsO))
        print("Point {} is expeted to be {} and was predicted as {}".format(point, xored[index],
                                                                            predict(point, weightsH, weightsO)[1]))


start_prog(1)
