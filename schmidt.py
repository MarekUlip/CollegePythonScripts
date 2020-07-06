import numpy as np

def schmidt(V):
    n = len(V)
    k = len(V[0])
    U = [[0 for j in range(k)] for i in range(n)]