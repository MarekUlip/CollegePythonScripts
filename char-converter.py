import numpy as np
def convert(array_to_convert):
    #array_to_convert = array_to_convert.ravel()
    print(array_to_convert)
    new_arr = []
    for _, item in enumerate(array_to_convert):
        if item == -1:
            new_arr.append(" ")
        else:
            new_arr.append("*")
    return np.array(new_arr)

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

base2 = np.array([-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,-1,1,1,-1,-1,-1,-1,
-1,-1,-1,1,1,1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,1,-1,-1,-1,
-1,-1,1,1,-1,-1,1,1,-1,-1,
-1,-1,1,-1,-1,-1,-1,1,-1,-1,
-1,1,1,1,1,1,1,1,1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

np.array([1,1,1,1,1,1,-1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,1,1,1,1,1,-1,-1,-1,
1,1,-1,-1,-1,1,1,1,-1,-1,
1,1,-1,-1,-1,-1,1,1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

base_B = convert(base_B)
print(np.reshape(base_B,(-1,1-1)))

