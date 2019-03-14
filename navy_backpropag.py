import math
import numpy as np

def sigm_(x):
    return x*(1-x)


def sigm(x):
    return 1/(1+np.exp(-x))

def count_error(expected, predicted):
    return expected - predicted

def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] for item in input])

def predict(point, bias, weightsH, weightsO):
    inpts = [[[point[0],weightsH[0]], [point[1],weightsH[1]]], [[point[0],weightsH[2]], [point[1],weightsH[3]]]]
    outsH = []
    outsH.append(sigm(dot_product(inpts[0], bias[0])))
    outsH.append(sigm(dot_product(inpts[1],bias[0])))
    inpts = [[outsH[0],weightsO[0]], [outsH[1],weightsO[1]]]
    out = sigm(dot_product(inpts,bias[1]))
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

def backpropagation(error, outO, outsH, weightsO, weightsH, learnConst, inputs):
    dO = error * sigm_(outO)
    dH = [dO*weightO*sigm_(outH) for weightO, outH in zip(weightsO, outsH)]
    weightsO = [weightO+learnConst*dO*outH for weightO, outH in zip(weightsO, outsH)]
    for i in [0,1]:
        weightsH[i] += learnConst*dH[0]*inputs[i % 2]
    for i in [2,3]:
        weightsH[i] += learnConst * dH[1] * inputs[i % 2]

    return weightsO, weightsH

def adapt_bias(bias, error, leaning_const):
    return bias + error*leaning_const


def start_prog(epochs):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    learn_const = 0.2
    weightsH = [np.random.uniform() for i in range(4)]
    weightsO = [np.random.uniform() for i in range(2)]
    biases = [1,1]
    for epoch in range(epochs):
        error_sum = 0
        for index, point in enumerate(inputs):
            outsH, out = predict(point, biases, weightsH, weightsO)
            error = count_error(xored[index], out)
            error_sum += math.fabs(error)
            weightsO, weightsH = backpropagation(error, out, outsH, weightsO, weightsH, learn_const, point)
        if error_sum < 0.5:
            print("{}: {}".format(epoch,error_sum))

    for index, point in enumerate(inputs):
        #print(predict(point, biases, weightsH, weightsO))
        print("Point {} is expeted to be {} and was predicted as {}".format(point, xored[index], predict(point, biases, weightsH, weightsO)[1]))


start_prog(25000)

