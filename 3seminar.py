import csv
"""import matplotlib.pyplot as plt
import numpy as np"""

# karate club
# 34 vrcholu

original_data = []

matrix = [[0 for col in range(34)] for row in range(34)]
with open('KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        original_data.append(row)
        col = int(row[1])-1
        row = int(row[0])-1
        matrix[row][col] = 1
        matrix[col][row] = 1

[print(row) for row in matrix]

def count_avg_degree(vert_degrees):
    total = sum(vert_degrees.values())
    print("Average degree: {}".format(total/len(vert_degrees)))
    """print(max(vert_degrees.values()))
    print(min(vert_degrees.values()))"""

def count_degrees_for_vertices(matrix):
    vert_degrees={}
    for index, row in enumerate(matrix):
        vert_degrees[index+1] = sum(row)
    print("Vertices degrees: {}".format(vert_degrees))
    return vert_degrees



count_avg_degree(count_degrees_for_vertices(matrix))




"""def count_vertice_degree(matrix):
    vertices = [sum([x for x in row]) for row in matrix]
    degree_count = {}
    for x in vertices:
        if x not in degree_count:
            degree_count[x] = 1
        else:
            degree_count[x]+=1
    print(degree_count)
    return degree_count"""

"""def min_max_and_avg(matrix):
    vertices = [sum([x for x in row]) for row in matrix]
    print(vertices)
    print(max(vertices))
    print(min(vertices))
    print(sum(x/2 for x in vertices)/len(vertices))

def show_histogram(matrix):
    degree_count = count_vertice_degree(matrix)
    names = ['Degree','Count']
    formats = ['f8','f8']
    dtype = dict(names = names, formats=formats)

    d = np.array(degree_count.items())
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.title('My Very Own Histogram')
    plt.grid(axis='y', alpha=0.75)


min_max_and_avg(matrix)
count_vertice_degree(matrix)
show_histogram(matrix)"""