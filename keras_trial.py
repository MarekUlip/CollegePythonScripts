from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np

#2.5, 3.3, 4.0
def create_train_set(a, count):
    base_x = np.random.uniform()
    train_set = []
    for i in range(count):
        train_set.append([base_x,a*base_x*(1-base_x)])
        base_x = train_set[i][1]
    return train_set



print(create_train_set(3.3,1000))