import numpy as np

base_A = np.array( [-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,1,1,1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,1,-1,-1,-1,
-1,-1,1,1,-1,-1,1,1,-1,-1,
-1,-1,1,-1,-1,-1,-1,1,-1,-1,
-1,1,1,1,1,1,1,1,1,-1,
-1,1,1,1,1,1,1,1,1,-1,
1,1,-1,-1,-1,-1,-1,-1,1,1,
1,1,-1,-1,-1,-1,-1,-1,1,1])

a_corrupted = np.array([-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,1,1,1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,1,-1,-1,-1,
-1,-1,1,1,-1,-1,1,1,-1,-1,
-1,-1,1,-1,-1,-1,-1,1,-1,-1,
-1,1,1,1,1,1,1,1,1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

a_corrupted_2 = np.array([-1,-1,-1,-1,1,1,-1,-1,-1,-1,
1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,-1,1,-1,-1,-1,
-1,-1,1,1,-1,-1,1,1,-1,-1,
-1,-1,1,-1,-1,-1,-1,1,-1,-1,
-1,1,-1,1,1,1,1,1,1,-1,
-1,1,-1,1,1,1,1,1,1,-1,
1,1,1,1,-1,-1,-1,-1,1,1,
1,1,1,-1,-1,-1,-1,-1,1,1])

b_corrupted = np.array([1,1,1,1,1,1,-1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,1,-1,-1,
1,1,-1,-1,-1,-1,1,1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

base_B = np.array([1,1,1,1,1,1,-1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,1,-1,-1,
1,1,-1,-1,-1,-1,1,1,-1,-1,
1,1,-1,-1,-1,1,1,1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,1,1,1,1,-1,-1,-1,-1])



patterns = np.array([base_A, base_B])

def convert(array_to_convert):
    new_arr = []
    for _, item in enumerate(array_to_convert):
        if item == -1:
            new_arr.append(" ")
        else:
            new_arr.append("*")
    return np.array(new_arr)

def noisify(p, array_to_noisify):
    for i in range(len(array_to_noisify)):
        rnd = np.random.uniform()
        if rnd > p:
            array_to_noisify[i] *= -1
    return array_to_noisify

def test(base, corrupted):
    print("-------------------------------------------------")
    print("Input shape:")
    char_variable = convert(corrupted)
    print_char(char_variable)
    print("expected shape:")
    char_variable = convert(base)
    print_char(char_variable)
    print("Recovered shape:")
    char_variable = convert(recover(corrupted))
    print_char(char_variable)
    print("-------------------------------------------------")

def print_char(char_variable):
    for row in np.reshape(char_variable, (-1, 10)):
        line = "".join(col for col in row)
        print(line)

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
    weight_matrix = create_weight_matrix()
    n = weight_matrix.shape[0]
    time_since_change = 0
    prev_random = np.random.randint(0,n)
    for i in range(1000):
        #print(time_since_change)
        rnd = np.random.randint(0,n)
        while rnd == prev_random:
            rnd = np.random.randint(0, n)
        prev_random = rnd
        col = weight_matrix[:, rnd]
        prev = pattern[rnd]
        pattern[rnd] = sigmod(np.dot(pattern,col))
        """if prev == pattern[rnd]:
            time_since_change += 1
        else:
            time_since_change = 0
        if time_since_change > 20:
            print("breaking")
            break"""
    #print("inpt: {}\noutput: {}".format(init_pattern,pattern))
    return pattern

test(base_A,a_corrupted)
test(base_A,a_corrupted_2)
test(base_A, noisify(0.7,base_A.copy()))
test(base_B,b_corrupted)
test(base_B,noisify(0.7,base_B.copy()))
#recover(np.array([-1,-1,1,-1]))

#weight_matrix = create_weight_matrix()