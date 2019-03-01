import matplotlib.pyplot as plt
import numpy as np

# correction is weight + Error*lambda
# lambda can be 0.1

def dot_product(input, bias):
    print(input)
    #print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0]*item[1] + bias for item in input])

def activation_function(input):
    if input < 0:
        return -1
    elif input == 0:
        return 0
    else:
        return 1

def count_error(expected, predicted):
    return expected - predicted

def adaptation(weight, error, inpt_val, learning_const):
    return weight + error*inpt_val*learning_const

def predict(input, bias):
    return activation_function(dot_product(input, bias))

def get_true_res(x, y):
    return int((2*x+1) > y)

def get_color(point, predicted):
    if get_true_res(point[0], point[1]) == predicted:
        return 'g'
    else:
        return 'r'

def train(num_of_points, learning_const):
    x_train = np.random.randint(-25, 25, num_of_points)
    y_train = np.random.randint(-25, 25, num_of_points)
    bias = np.ones(num_of_points)
    weights = np.random.uniform(size=2)

    for i in range(num_of_points):
        point = [[x_train[i], weights[0]], [y_train[i], weights[1]]]
        res = predict(point, bias)
        true_res = get_true_res(x_train[i], y_train[i])
        if res != true_res:
            error = count_error(true_res, res)
            weights[0] = adaptation(weights[0], error, x_train[i], learning_const)
            weights[1] = adaptation(weights[1], error, x_train[i], learning_const)

    x_test = np.random.randint(-25, 25, num_of_points)
    y_test = np.random.randint(-25, 25, num_of_points)


    colors = []

    for i in range(num_of_points):
        point = [[x_test[i], weights[0]], [y_test[i], weights[1]]]
        res = predict(point, bias)
        colors.append(get_color([x_test[i],y_test[i]],res))

    ll = np.arange(-25,25,1)
    plt.plot(ll, 2*ll + 1)
    plt.scatter(x_test, y_test, colors)
    plt.grid(True)
    plt.show()

train(25, 0.1)