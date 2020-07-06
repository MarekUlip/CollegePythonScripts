import csv
import sys

start_matrix = [[0 for x in range(24)] for y in range(10)]
start_matrix2 = [[0 for x in range(10)] for y in range(24)]
with open('with_gere.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        print(row)
        row[0] = int(row[0])
        row[1] = int(row[1])
        col = int(row[1])-1-24
        row = int(row[0])-1
        start_matrix[row][col] = 1

with open('with_gere.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        print(row)
        row[0] = int(row[0])
        row[1] = int(row[1])
        roww = int(row[1])-1-24
        col = int(row[0])-1
        start_matrix2[roww][col] = 1

test_matrix = [[1,1,1,0,0,0,0],
                [0,1,1,1,1,0,0],
                [0,0,0,1,0,1,0],
                [0,0,0,0,1,1,1]]

def create_partity_of_y(matrix):
    cols = len(matrix[0])
    rows = len(matrix)
    new_matrix = [[0 for x in range(rows)] for x in range(rows)]
    for j in range(cols):
        for i in range(rows):
            if matrix[i][j] == 1:
                for k in range(i,rows):
                    if matrix[k][j] == 1:
                        new_matrix[i][k] = new_matrix[k][i] = 1
    return new_matrix

def create_partity_of_x(matrix):
    cols = len(matrix[0])
    rows = len(matrix)
    new_matrix = [[0 for x in range(cols)] for x in range(cols)]
    for j in range(cols):
        for i in range(rows):
            if matrix[i][j] == 1:
                for k in range(i,rows):
                    if matrix[k][j] == 1:
                        for l in range(cols):
                            if matrix[i][l] == 1 or matrix[k][l] == 1:
                                new_matrix[j][l] = new_matrix[l][j] = 1
    return new_matrix

#[print(row) for row in create_partity_of_x(test_matrix)]
#[print(row) for row in create_partity_of_y(test_matrix)]
def prep_matrix(matrix):
    matrix = matrix.copy()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i == j: 
                    continue
                matrix[i][j] = sys.float_info.max
    return matrix

#matrix = prep_matrix(start_matrix)
def floyd_algorithm(matrix):
    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
    return matrix

def count_avg_degree(vert_degrees):
    total = sum(vert_degrees.values())
    return total/len(vert_degrees)

def count_degrees_for_vertices(matrix):
    vert_degrees={}
    for index, row in enumerate(matrix):
        vert_degrees[index+1] = sum(row)-1
    return vert_degrees

def find_diameter(matrix):
    #print(matrix)
    biggest = 0
    for row in matrix:
        for col in row:
            if col != 0:
                if col > biggest:
                    biggest = col
    return biggest

movies = create_partity_of_y(start_matrix)
[print(row) for row in movies]
degrees = count_degrees_for_vertices(movies)
avg_degrees = count_avg_degree(degrees)
diameter = find_diameter(floyd_algorithm(prep_matrix(movies)))
print("actors")
print("Degrees: {}\nAvg degrees: {}\nDiameter: {}\n".format(degrees,avg_degrees,diameter))

actors = create_partity_of_x(start_matrix)
a1 = actors[10:]
a2 = actors[:10]
actors = a1+a2
[print(row) for row in actors]
degrees = count_degrees_for_vertices(actors)
avg_degrees = count_avg_degree(degrees)
diameter = find_diameter(floyd_algorithm(prep_matrix(actors)))
print("movies")
print("Degrees: {}\nAvg degrees: {}\nDiameter: {}\n".format(degrees,avg_degrees,diameter))

[print(row) for row in start_matrix]