import math
import operator

matrix = [[2,1,-1,8], [-3,-1,2,-11], [-2,1,2,-3]]

def find_pivot_for_col(matrix,start_row,col_num):
    max_num_row = start_row
    for i in range(start_row+1,len(matrix)):
        if math.fabs(matrix[i][col_num]) > math.fabs(matrix[max_num_row][col_num]):
            max_num_row = i
    if max_num_row != start_row:
        matrix[start_row], matrix[max_num_row] = matrix[max_num_row], matrix[start_row]
    return matrix[start_row][col_num]


def gausian_elimination(matrix):
    h = len(matrix)
    w = len(matrix[0])
    i = 0
    j = 0
    while i < h and j < w:
        col_val = find_pivot_for_col(matrix,i,j)
        if col_val == 0:
            j+=1
            continue
        else:
            for k in range(i+1,h):
                r_mult = matrix[k][j] / matrix[i][j] 
                matrix[k][i] = 0
                for l in range(i+1,w):
                    matrix[k][l] -= (matrix[i][l] * r_mult)
            i+=1
            j+=1

def gausian_wiki(A):
    m = len(A)
    n = len(A[0])
    h = 0
    k = 0
    while h < m and k < n:
        index, value = max(enumerate(math.fabs(row[k]) for row in A[h:]), key=operator.itemgetter(1)) #max(math.fabs(row[k]) for row in A)
        #print(i_max)
        if value == 0:
            k += 1
        else:
            A_help = A[h]
            A[h] = A[index]
            A[index] = A_help

            for i in range(h+1,m):
                f = A[i][k] / A[h][k]
                A[i][k] = 0
                
                for j in range(k+1, n):
                    A[i][j] = A[i][j] - A[h][j] *f
            h += 1
            k += 1
    matrix.reverse()

def gausian_elimination2(matrix):
    h = len(matrix)
    w = len(matrix[0])

    for i in range(h):
        max_row_num = i
        max_val= math.fabs(matrix[i][i])
        for j in range(i,h):
            if math.fabs(matrix[j][i]) > max_val:
                max_row_num = j
                max_val = math.fabs(matrix[j][i])
        
        if max_row_num != i:
            matrix_help = matrix[i]
            matrix[i] = matrix[max_row_num]
            matrix[max_row_num] = matrix_help
        
        for j in range(i+1,h):
            row_mult = matrix[j][i] / matrix[i][i]
            matrix[j][i] = 0
            for k in range(i+1,w):
                matrix[j][k] -= matrix[i][k]*row_mult

def gauss(A):
    n = len(A)

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        #for k in range(i-1, -1, -1):
            #A[k][n] -= A[k][i] * x[i]
    return x

gausian_elimination(matrix)
#gausian_wiki(matrix)
#print(gauss(matrix))
#gausian_elimination2(matrix)
[print(row) for row in matrix]