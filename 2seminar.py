import numpy as np
import time

def generate_random_sequence(seq_len):
    alphabet = ["A","C","G","T"]
    sequence = ""
    for _ in range(seq_len):
        sequence+=alphabet[np.random.randint(4)]
    return sequence
print(generate_random_sequence(10))

def hamming_distance(seq_a, seq_b):
    min_len = min([len(seq_a),len(seq_b)])
    distance = 0
    for i in range(min_len):
        if seq_a[i] != seq_b[i]:
            distance += 1
    len_diff = np.abs(len(seq_a)-len(seq_b))
    return distance + len_diff

#print(hamming_distance("ABCE","ABC"))

def editational_distance(seq_a, len_a, seq_b, len_b):
    #len_a = len(seq_a)
    #len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    cost = 0
    if seq_a[len_a-1] != seq_b[len_b-1]:
        cost = 1
    
    return min([editational_distance(seq_a,len_a-1,seq_b,len_b)+1,editational_distance(seq_a,len_a,seq_b,len_b-1)+1,editational_distance(seq_a,len_a-1,seq_b,len_b-1)+cost])

def seqences_matches(seq_a, index_a,seq_b,index_b):
    cost = 0
    if seq_a[index_a] != seq_b[index_b]:
        cost = 1
    return cost

def dynamic_editational_distance(seq_a,seq_b):
    len_a = len(seq_a)+1
    len_b = len(seq_b)+1
    matrix = np.zeros((len_a,len_b),dtype=int)
    for i in range(1,len_a):
        matrix[i,0] = matrix[i-1,0]+1
    for i in range(1,len_b):
        matrix[0,i] = matrix[0,i-1]+1
    
    for i in range(1,len_a):
        for j in range(1,len_b):
            matrix[i,j] = min([matrix[i-1,j]+1, matrix[i,j-1]+1, matrix[i-1,j-1] + seqences_matches(seq_a,i-1,seq_b,j-1)])
    return matrix[len_a-1,len_b-1]
    
#n = 5
for n in range(1,1001):
    print("Sequence length: {}".format(n))
    seq_a = generate_random_sequence(n)
    seq_b = generate_random_sequence(n)
    #print("{}\n{}".format(seq_a,seq_b))
    start = time.time()
    print(editational_distance(seq_a,n,seq_b,n))
    print("Recurrent editational took {}s".format(time.time()-start))
    start = time.time()
    print(dynamic_editational_distance(seq_a,seq_b))
    print("Dynamic editational took {}s".format(time.time()-start))