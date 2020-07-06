import math
import random
import csv
import sys

edges = []
def create_random_graph(n, p):
    # p by melo by vetsi nez 0,000921 pro 10000 vrcholu  - ostry prah nad kterym bude sit souvisla
    matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            col = j+i+1
            if col >= n:
                break
            rnd = random.random()
            if rnd < p:
                matrix[i][col] = 1
                edges.append([i,col])
                matrix[col][i] = 1
    #[print(row) for row in matrix]
    write_vertices_to_csv(edges,"random-chart.csv")
    return matrix

def count_min_probabilty(n):
    return math.log(n)/n

def write_vertices_to_csv(edges,name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)

def count_avg_degree(vert_degrees):
    total = sum(vert_degrees.values())
    #print("Average degree: {}".format(total/len(vert_degrees)))
    return total/len(vert_degrees)
    """print(max(vert_degrees.values()))
    print(min(vert_degrees.values()))"""

def count_degrees_for_vertices(matrix):
    vert_degrees={}
    for index, row in enumerate(matrix):
        vert_degrees[index+1] = sum(row)
    #print("Vertices degrees: {}".format(vert_degrees))
    return vert_degrees

def count_closeness_centrality(matrix):
    closeness = []
    n = len(matrix)
    for i in range(n):
        closeness.append(0)
        for j in range(len(matrix[i])):
            closeness[i] = n/sum([x for x in matrix[i]])
    return closeness

def clustering_koeficient(vertice,matrix):
    vertice_neigh = []
    for index, item in enumerate(matrix[vertice-1]):
        if item == 1:
            vertice_neigh.append(index+1)
    neighbours = sum([x for x in matrix[vertice-1]])
    vertex = []
    for vert in vertice_neigh:
        for vert2 in vertice_neigh:
            if vert == vert2 or ((vert2, vert) in vertex):
                continue
            vertex.append((vert,vert2))
    num_of_edges = 0
    for v in vertex:
        if matrix[v[0]-1][v[1]-1] == 1:
            num_of_edges += 1
    if neighbours*(neighbours-1) == 0:
        return -1
    clstr_koef = (2*num_of_edges)/(neighbours*(neighbours-1))
    return clstr_koef

def transform_matrix_for_floyd(matrix):
    dim = len(matrix)
    for i in range(dim):
        for j in range(dim):
            if matrix[i][j] == 0:
                matrix[i][j] = sys.float_info.max

def prep_matrix(matrix):
            
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i == j: 
                    continue
                matrix[i][j] = sys.float_info.max
    return matrix

def floyd_algorithm(matrix):
    #print(matrix)
    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
    return matrix

def find_diameter(matrix):
    #print(matrix)
    biggest = 0
    for row in matrix:
        for col in row:
            if col != 0:
                if col > biggest:
                    biggest = col
    return biggest

def create_barabasi_albert_graph_dynamic(n,m,m0):
    edges = []
    vertice_count = n-m0
    current_count = m0
    vertice_list = []
    for i in range(m0):
        for j in range(m0-1):
            vertice_list.append(i)
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(m0-1):
        for j in range(1,m0):
            if i == j:
                continue
            edges.append([i,j])
            neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1
    print(edges)
    print(vertice_list)
    for i in range(vertice_count):
        v_neighs = []
        for j in range(m):
            rnd = random.randint(0,len(vertice_list)-1)
            if vertice_list[rnd] not in v_neighs:
                v_neighs.append(vertice_list[rnd])
        for neigh in v_neighs:
            vertice_list.insert(vertice_list.index(neigh),neigh)
            edges.append([current_count,neigh])
            neighbour_matrix[current_count][neigh] = neighbour_matrix[neigh][current_count] = 1
        for j in range(m):
            vertice_list.append(current_count)
        current_count+=1
    write_vertices_to_csv(edges,"barabasi-albert.csv")
    print("Vertice list: {} {}".format(vertice_list, vertice_count))
    return neighbour_matrix

def create_barabasi_albert_graph(n,m):
    edges = []
    m0 = 3
    vertice_count = n-m0
    current_count = m0
    vertice_list = [0,0,1,1,2,2]
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    edges.append([0,1])
    edges.append([0,2])
    edges.append([1,2])
    neighbour_matrix[0][1] = neighbour_matrix[1][0] = 1
    neighbour_matrix[2][1] = neighbour_matrix[1][2] = 1
    neighbour_matrix[0][2] = neighbour_matrix[2][0] = 1
    for i in range(vertice_count-1):
        current_count+=1
        v_neighs = []
        for j in range(m):
            rnd = random.randint(0,len(vertice_list)-1)
            if vertice_list[rnd] not in v_neighs:
                v_neighs.append(vertice_list[rnd])
        for neigh in v_neighs:
            vertice_list.insert(vertice_list.index(neigh),neigh)
            edges.append([current_count,neigh])
            neighbour_matrix[current_count][neigh] = neighbour_matrix[neigh][current_count] = 1
        for j in range(m):
            vertice_list.append(current_count)
    write_vertices_to_csv(edges,"barabasi-albert.csv")
    return neighbour_matrix
    
def count_average_clustering_coeficient(n,matrix):
    clusters = []
    distribution = []
    for i in range(1,n):
        res = clustering_koeficient(i,matrix)
        if res == -1:
            continue
        clusters.append(res)
        distribution.append([i,sum([x for x in matrix[i-1]]), res])
        #print("{}. {}".format(i,clustering_koeficient(i)))
    return clusters,distribution

def count_average_clustering_coeficient_distribution(distribution):
    avg_res = {}
    for vertice in distribution:
        if vertice[1] not in avg_res:
            avg_res[vertice[1]] = [1, vertice[2]]
        else:
            avg_res[vertice[1]] = [avg_res[vertice[1]][0]+1, avg_res[vertice[1]][1] + vertice[2]]

    for key, item in sorted(avg_res.items()):
        print("Degree {} has average clustering coeficient {}".format(key,item[1]/item[0]))
    

n=500
p = 0.012429216196844383 * 2#0.046*2 #04605170185988092
matrix = create_random_graph(n,p)
avg_degree = count_avg_degree(count_degrees_for_vertices(matrix))
matrix2 = create_barabasi_albert_graph_dynamic(n,3,10)
avg_degree_ba = count_avg_degree(count_degrees_for_vertices(matrix2))

print("Average degree: {}. Average degree based on random graph: {}".format(avg_degree, (n-1)*p))
print("Average degree B-A graph: {}".format(avg_degree_ba))
    
print(count_min_probabilty(n))

clusters, distribution = count_average_clustering_coeficient(n,matrix)
print("Average clustering coeficient for random graph is {}".format(sum([x for x in clusters])/len(clusters)))

print("Average clustering coeficient per degree for random graph")
count_average_clustering_coeficient_distribution(distribution)


clusters, distribution = count_average_clustering_coeficient(n,matrix2)
print("Average clustering coeficient for B-A graph is {}".format(sum([x for x in clusters])/len(clusters)))

print("Average clustering coeficient per degree for B-A graph")
count_average_clustering_coeficient_distribution(distribution)

m = matrix.copy()
m = prep_matrix(m)
print("Matrix_prepared")
m = floyd_algorithm(m)
print("Diameter for random graph is {}".format(find_diameter(m)))

m = matrix2.copy()
m = prep_matrix(m)
m = floyd_algorithm(m)
print("Diameter for B-A is {}".format(find_diameter(m)))

