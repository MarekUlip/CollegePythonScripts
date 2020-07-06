import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter import tix
from tkinter import ttk

# correction is weight + Error*lambda
# lambda can be 0.1

root = tix.Tk()
alg_params = {
    "min_val": StringVar(value="-25"),
    "max_val": StringVar(value="25"),
    "num_of_train_p": StringVar(value="100"),
    "num_of_test_p": StringVar(value="100"),
    "epochs": StringVar(value="10"),
    "bias": StringVar(value="1"),
    "learning_const": StringVar(value="0.1")
}


def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] + bias for item in input])


def activation_function(input):
    if input < 0:
        return 0
    # elif input == 0:
    #    return 0
    else:
        return 1


def count_error(expected, predicted):
    return expected - predicted


def adaptation(weight, error, inpt_val, learning_const):
    return weight + error * inpt_val * learning_const


def adapt_bias(bias, error, leaning_const):
    return bias + error * leaning_const


def predict(input, bias):
    return activation_function(dot_product(input, bias))


def get_true_res(x, y):
    return int((2 * x + 1) > y)


def get_color(point, predicted):
    print("True result: {}. Predicted {}".format(get_true_res(point[0], point[1]), predicted))
    if get_true_res(point[0], point[1]) == predicted:
        return 'g'
    else:
        return 'r'


def get_point_pos_color(point):
    res = 2 * point[0] + 1 - point[1]
    if res > 0:
        return 'black'
    elif res == 0:
        return 'yellow'
    else:
        return 'cyan'


def train(num_of_train_points, num_of_test_points, bias, epochs, learning_const, min_val, max_val):
    train_points_count = num_of_train_points
    x_train = [np.random.randint(min_val, max_val) for i in
               range(train_points_count)]  # np.random.randint(min_val, max_val, num_of_points)
    y_train = [np.random.randint(min_val, max_val) for i in range(train_points_count)]
    bias = bias
    weights = [np.random.uniform() for i in range(2)]

    for epoch in range(epochs):
        for i in range(train_points_count):
            # print(weights)
            point = [[x_train[i], weights[0]], [y_train[i], weights[1]]]
            res = predict(point, bias)
            true_res = get_true_res(x_train[i], y_train[i])
            if res != true_res:
                error = count_error(true_res, res)
                weights[0] = adaptation(weights[0], error, x_train[i], learning_const)
                weights[1] = adaptation(weights[1], error, y_train[i], learning_const)
                bias = adapt_bias(bias, error, learning_const)

    x_test = [np.random.randint(min_val, max_val) for i in range(num_of_test_points)]
    y_test = [np.random.randint(min_val, max_val) for i in range(num_of_test_points)]

    colors = []
    true_colors = []

    for i in range(num_of_test_points):
        point = [[x_test[i], weights[0]], [y_test[i], weights[1]]]
        res = predict(point, bias)
        colors.append(get_color([x_test[i], y_test[i]], res))
        true_colors.append(get_point_pos_color([x_test[i], y_test[i]]))

    ll = np.arange(min_val, max_val, 1)
    plt.plot(ll, 2 * ll + 1)
    plt.scatter(x_test, y_test, c=true_colors, edgecolors=colors)
    plt.grid(True)
    plt.show()


def start_alg():
    train(int(alg_params["num_of_train_p"].get()),
          int(alg_params["num_of_test_p"].get()),
          float(alg_params["bias"].get()),
          int(alg_params["epochs"].get()),
          float(alg_params["learning_const"].get()),
          int(alg_params["min_val"].get()),
          int(alg_params["max_val"].get()))


if True:
    font_size = 32

    main_frame = Frame(root)
    main_frame.pack(fill=BOTH)
    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Perceptron", font=("Arial", 44)).pack(anchor='center')
    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["min_val"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["max_val"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Num of train points", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["num_of_train_p"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Num of test points", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["num_of_test_p"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Epochs", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["epochs"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Bias", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["bias"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=X)
    Label(wrapper, text="Learning Constant", font=("Arial", font_size)).pack(side=LEFT)
    Entry(wrapper, textvariable=alg_params["learning_const"], font=("Arial", font_size)).pack(side=RIGHT)

    wrapper = Frame(main_frame)
    wrapper.pack(fill=BOTH)
    Button(wrapper, command=start_alg, text="Start", font=("Arial", font_size)).pack()

    root.mainloop()

# train(1000, 0.1)
