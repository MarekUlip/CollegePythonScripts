import math
import numpy as np

def sigm_(x):
    return x*(1-x)


def sigm(x):
    return 1/(1-math.e**(-x))

def count_error(expected, predicted):
    return expected - predicted

def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] + bias for item in input])

def predict(point, bias, weightsH, weightsO):
    inpts = [[[point[0],weightsH[0]], [point[1],weightsH[1]]], [[point[0],weightsH[2]], [point[1],weightsH[3]]]]
    outsH = []
    outsH.append(sigm(dot_product(inpts[0], bias[0])))
    outsH.append(sigm(dot_product(inpts[1],bias[0])))
    inpts = [[outsH[0],weightsO[0]], [outsH[1],weightsO[1]]]
    out = sigm(dot_product(inpts,bias[1]))
    return outsH, out



def backpropagation(error, outO, outsH, weightsO, weightsH, learnConst, inputs):
    dO = error * sigm_(outO)
    dH = [dO*weightO*sigm_(outH) for weightO, outH in zip(weightsH, outsH)]
    bias = [1,1]
    weightsO = [weightO*learnConst*dO*outH for weightO, outH in zip(weightsO, outsH)]
    weightsH = [weightH*learnConst*dH*inpt for weightH, inpt in zip(weightsH,inputs[0])]
    #weightsH = [weightH*learnConst*dH*inpt for weightH, inpt in zip(weightsH,inputs)]


def start_prog(epochs):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    learn_const = 0.1
    weightsH = [np.random.uniform() for i in range(4)]
    weightsO = [np.random.uniform() for i in range(2)]
    biases = [1,1]
    for epoch in range(epochs):
        for index, point in enumerate(inputs):
            print("*")
            outsH, out = predict(point, biases, weightsH, weightsO)
            error = count_error(xored[index], out)
            backpropagation(error, out, outsH, weightsO, weightsH, learn_const, point)

    for index, point in enumerate(inputs):
        print("Point {} is expeted to be {} and was predicted as {}".format(point, xored[index], predict(point, biases, weightsH, weightsO)[1]))


start_prog(10)

