import numpy as np

base_A = np.array( [0,0,0,0,1,1,0,0,0,0,
0,0,0,0,1,1,0,0,0,0,
0,0,0,1,1,1,1,0,0,0,
0,0,0,1,0,0,1,0,0,0,
0,0,1,1,0,0,1,1,0,0,
0,0,1,0,0,0,0,1,0,0,
0,1,1,1,1,1,1,1,1,0,
0,1,1,1,1,1,1,1,1,0,
1,1,0,0,0,0,0,0,1,1,
1,1,0,0,0,0,0,0,1,1])

base_B = np.array([1,1,1,1,1,1,0,0,0,0,
1,1,1,1,1,1,1,0,0,0,
1,1,0,0,0,1,1,0,0,0,
1,1,1,1,1,1,1,0,0,0,
1,1,1,1,1,1,1,0,0,0,
1,1,0,0,0,1,1,1,0,0,
1,1,0,0,0,0,1,1,0,0,
1,1,0,0,0,1,1,1,0,0,
1,1,1,1,1,1,1,0,0,0,
1,1,1,1,1,1,0,0,0,0])

patterns = np.array([[1,-1,1,-1]])

def sigmod(x, treshold = 0):
    if x >= treshold:
        return 1
    else:
        return -1


def create_weight_matrix():
    base = np.dot(patterns.transpose(), patterns)
    sigm= np.vectorize(sigmod)
    base = sigm(base)
    return base - np.identity(patterns.shape[1])

def recover(pattern):
    init_pattern = pattern.copy()
    weight_matrix = create_weight_matrix()
    print(weight_matrix)
    n = weight_matrix.shape[0]
    time_since_change = 0
    for i in range(100):
        rnd = np.random.randint(0,n)
        col = weight_matrix[:, rnd]
        prev = pattern[rnd]
        pattern[rnd] = sigmod(np.dot(pattern,col))
        if prev == pattern[rnd]:
            time_since_change += 1
        else:
            time_since_change = 0
        if time_since_change > 20:
            break
    print("inpt: {}\noutput: {}".format(init_pattern,pattern))
    return pattern

recover(np.array([-1,-1,1,-1]))

#weight_matrix = create_weight_matrix()