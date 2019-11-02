import numpy as np


pairs = {'A':'U','U':'A','C':'G','G':'C'}
def initialize(n):
    n=n
    return np.zeros((n,n),dtype='int')

def find_max_for_range_complementary(sequence, matrix, i,j):
    results = [0]
    for k in range(i,j):
        if is_complementary(sequence[k],sequence[j]):
            results.append(matrix[i,k-1]+matrix[k+1,j-1]+1)
    print(results)
    return max(results)

def find_max_for_range(sequence, matrix, i,j):
    results = []
    for k in range(i,j):
        results.append(matrix[i,k]+matrix[k+1,j])
    print(results)
    return max(results)

def is_complementary(a,b,pairs=pairs):
    if pairs[a] == b:
        return True
    return False

def cost_func(a,b,pairs=pairs):
    if is_complementary(a,b,pairs):
        return 1
    return 0

def count_nussinov(sequence, matrix):
    n = len(sequence)
    for j in range(n):
        for i in range(n):
            if i == j or i-j > 0:
                continue
            matrix[i,j] = max([matrix[i+1,j-1]+cost_func(sequence[i],sequence[j]),find_max_for_range(sequence,matrix,i,j)])
    print(matrix)

sequence = 'ACCAGCU'
matrix = initialize(len(sequence))
count_nussinov(sequence,matrix)