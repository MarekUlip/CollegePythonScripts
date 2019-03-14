import csv
import math

matrix = [[0 for col in range(34)] for row in range(34)]
nodes = []
with open('1KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        row[0] = int(row[0])
        row[1] = int(row[1])
        #original_data.append(row)
        col = int(row[1])-1
        row = int(row[0])-1
        matrix[row][col] = 1
        matrix[col][row] = 1
        nodes.append(row)

def count_similarity(matrix):
    sim_matrix = [[0 for col in range(34)] for row in range(34)]
    n = len(matrix)
    for i in range(n):
        a_nodes = []
        shared_nodes = []
        for index, col in enumerate(matrix[i]):
            if col > 0:
                a_nodes.append(index)
        for j in range(n):
            if i == j:
                continue
            b_nodes = []
            shared_nodes = []
            for index, col in enumerate(matrix[j]):
                if col > 0:
                    b_nodes.append(index)
                    if index in a_nodes:
                        shared_nodes.append(index)
            sim_matrix[i][j] = len(shared_nodes)/ math.sqrt(len(a_nodes)*len(b_nodes))
    return sim_matrix  

def clustering(matrix):
    clustered_nodes = []
    clusters = []
    comunity_size = 0
    while comunity_size < len(matrix):
        a, b = simple_linkage(matrix)
        if a in clustered_nodes:

def go_deeper(depth, cluster, element):
    if all(isinstance(x,float) for x in cluster):
        return [depth, element in cluster]
    else:
        if isinstance(cluster[1],float):
            return [depth, cluster[1] == element]
        go_deeper(depth+1, cluster[0],element)
        return go_deeper(depth+1, )

def find_biggest_cluster_with_element(element, clusters):
    for cluster in clusters:
        depth = 0
        while not all(isinstance(x,float), cluster):
            depth+=1

        

def simple_linkage(matrix):
    closest = [max(row) for row in matrix]
    max_val = max(closest)
    vertex_num = closest.index(max_val)
    sec_vertex_num = matrix[vertex_num].index(max_val)
    return [vertex_num, sec_vertex_num]

similarity_matrix = count_similarity(matrix)
similarity_matrix_work = similarity_matrix.copy()
simple_linkage(similarity_matrix)      
#[print(row) for row in count_similarity(matrix)]

    