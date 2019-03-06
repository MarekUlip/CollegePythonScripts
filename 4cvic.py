import random
import csv

def create_barabasi_albert_graph_dynamic(n,m,m0,p):
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
    #print(edges)
    #print(vertice_list)
    for i in range(vertice_count):
        v_neighs = []
        preferential_connection(vertice_list, v_neighs)
        rnd = v_neighs[0]
        """rnd = random.randint(0,len(vertice_list)-1)
        while vertice_list[rnd] in v_neighs:
            rnd = random.randint(0,len(vertice_list)-1)
        v_neighs.append(vertice_list[rnd])"""
        for j in range(m-1):
            probability = random.random()
            if p < probability:
                available_vertexes = get_indexes_of_vertex(neighbour_matrix,rnd,v_neighs)
                if len(available_vertexes)==1:
                    rnd_vertex = 0
                elif len(available_vertexes) == 0:
                    preferential_connection(vertice_list, v_neighs)
                    vertice_count += 1
                    continue
                else:
                    rnd_vertex = random.randint(0,len(available_vertexes)-1)
                v_neighs.append(available_vertexes[rnd_vertex])
                neighbour_matrix[vertice_count][available_vertexes[rnd_vertex]] = neighbour_matrix[available_vertexes[rnd_vertex]][vertice_count] = 1
            else:
                preferential_connection(vertice_list, v_neighs)
                """rnd = random.randint(0,len(vertice_list)-1)
                while vertice_list[rnd] in v_neighs:
                    rnd = random.randint(0,len(vertice_list)-1)
                v_neighs.append(vertice_list[rnd])"""
        for neigh in v_neighs:
            vertice_list.insert(vertice_list.index(neigh),neigh)
            edges.append([current_count,neigh])
            neighbour_matrix[current_count][neigh] = neighbour_matrix[neigh][current_count] = 1
        for j in range(m):
            vertice_list.append(current_count)
        current_count+=1
    #write_vertices_to_csv(edges,"barabasi-albert.csv")
    #print("Vertice list: {} {}".format(vertice_list, vertice_count))
    return neighbour_matrix

def preferential_connection(vertice_list, v_neighs):
    rnd = random.randint(0,len(vertice_list)-1)
    while vertice_list[rnd] in v_neighs:
        rnd = random.randint(0,len(vertice_list)-1)
    v_neighs.append(vertice_list[rnd])


def biancony_model(p,m,m0,n):
    vertice_count = 2
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    neighbour_matrix[0][1] = neighbour_matrix[1][0] = 1
    for i in range(m0-2):
        rnd = random.randint(0,vertice_count-1)
        #print("rnd:{} v_c:{}".format(rnd,vertice_count+i))
        neighbour_matrix[vertice_count][rnd] = neighbour_matrix[rnd][vertice_count] = 1
        vertice_count+=1
    for i in range(n-m0):
        rnd = random.randint(0, vertice_count-1)
        print("rnd: {}, v_c: {}".format(rnd, vertice_count))
        neighbour_matrix[vertice_count][rnd] = neighbour_matrix[rnd][vertice_count] = 1
        exclude = [vertice_count]
        for j in range(m-1):
            chance = random.random()
            if chance < p:
                available_vertexes = get_indexes_of_vertex(neighbour_matrix,rnd,exclude)
                if len(available_vertexes)==1:
                    rnd_vertex = 0
                elif len(available_vertexes) == 0:
                    random_connection(neighbour_matrix,exclude,vertice_count)
                    #vertice_count += 1
                    continue
                else:
                    rnd_vertex = random.randint(0,len(available_vertexes)-1)
                exclude.append(available_vertexes[rnd_vertex])
                neighbour_matrix[vertice_count][available_vertexes[rnd_vertex]] = neighbour_matrix[available_vertexes[rnd_vertex]][vertice_count] = 1
            else:
                random_connection(neighbour_matrix,exclude,vertice_count)
                """rnd = random.randint(0, vertice_count-1)
                while rnd in exclude:
                    print("Attempt to connect to already connected. Trying again.")
                    rnd = random.randint(0, vertice_count-1)
                neighbour_matrix[i+vertice_count][rnd] = neighbour_matrix[rnd][i+vertice_count] = 1
                exclude.append(rnd)"""
        vertice_count += 1
    return neighbour_matrix

def random_connection(neighbour_matrix, exclude, new_v_num):
    rnd = random.randint(0, new_v_num-1)
    while rnd in exclude:
        print("Attempt to connect to already connected. Trying again.")
        rnd = random.randint(0, new_v_num-1)
    neighbour_matrix[new_v_num][rnd] = neighbour_matrix[rnd][new_v_num] = 1
    exclude.append(rnd)

def get_indexes_of_vertex(neighbour_matrix, vertex, exclude):
    indexes = []
    for index, col in enumerate(neighbour_matrix[vertex]):
        if col > 0 and index not in exclude:
            indexes.append(index)
    return indexes

def safe_matrix_as_csv(matrix, name):
    edges = []
    #print(sum([sum(row) for row in matrix]))
    for index, row in enumerate(matrix):
        for col_index, col in enumerate(row):
            if col == 1:
                if [index, col_index] in edges or [col_index, index] in edges:
                    #print("Yea")
                    continue
                edges.append([index, col_index])
    #print(len(edges))
    write_vertices_to_csv(edges,"{}.csv".format(name))

def write_vertices_to_csv(edges,name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)

safe_matrix_as_csv(biancony_model(0.50,3,8,1000),"biancony3")
#safe_matrix_as_csv(create_barabasi_albert_graph_dynamic(1000,3,8,0.98),"holme")