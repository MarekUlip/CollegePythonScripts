"""from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D"""
import math
import numpy as np
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD


def test(a, forced, use_chaos):
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
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model.fit([train_set], results, batch_size=10, epochs=300)

    if use_chaos:
        train_set, results = create_test_set_chaos(a, 1000)
    else:
        train_set, results = create_test_set(a, 1000)

    max_diff = 0
    avg_difference = 0
    predicts = model.predict_proba([train_set])
    for index, point in enumerate(predicts):
        # print(predict(point, biases, weightsH, weightsO))
        predicted = point[0]  # predict(point, biases, weightsH, weightsO)[1]
        difference = results[index] - predicted
        avg_difference += math.fabs(difference)
        if math.fabs(difference) > max_diff:
            max_diff = math.fabs(difference)
        print("Point {} is expeted to be {} and was predicted as {}. Difference is {}".format(train_set[index],
                                                                                              results[index], predicted,
                                                                                              difference))
    print("Max difference was {}. Average difference was: {}".format(max_diff, avg_difference / len(train_set)))
    # print(model.predict_proba(X))


# 2.5, 3.3, 4.0
def create_train_set_write(a, count):
    base_x = np.random.uniform()
    train_set = []
    for i in range(count):
        train_set.append([base_x, a * base_x * (1 - base_x)])
        base_x = train_set[i][1]
    return train_set


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


test(4.0, False, True)
# print(create_train_set_write(4.0,1000))
