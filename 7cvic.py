import csv
import math
import copy
import sys
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import os


dimesion = 500
matrix = [[0 for col in range(dimesion)] for row in range(dimesion)]
nodes = []
edges = []
with open('1USairport500.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        edges.append(row.copy())
        row[0] = int(row[0])
        row[1] = int(row[1])
        #original_data.append(row)
        col = int(row[1])-1
        row = int(row[0])-1
        matrix[row][col] = 1
        matrix[col][row] = 1
        nodes.append(row)

def load_saved_graph(dimension, path = '1USairport500.csv'):
    matrix = [[0 for col in range(dimension)] for row in range(dimension)]
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            edges.append(row.copy())
            row[0] = int(row[0])
            row[1] = int(row[1])
            #original_data.append(row)
            col = int(row[1])-1
            row = int(row[0])-1
            matrix[row][col] = 1
            matrix[col][row] = 1
            nodes.append(row)
    return matrix

def create_barabasi_albert_graph_dynamic(n,m,m0, vertice_list = None):
    """
    Updated version that allows to set vertice list instead of creating full graph. Also fixed varing num of edges
    """
    edges = []
    vertice_count = n-m0
    current_count = m0
    if vertice_list is None:
        vertice_list = []
        for i in range(m0):
            for j in range(m0-1):
                vertice_list.append(i)
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(m0):
        for j in range(m0):
            if i == j:
                continue
            edges.append([i,j])
            neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1
    #print(edges)
    #print(vertice_list)
    for i in range(vertice_count):
        v_neighs = []
        for j in range(m):
            rnd = random.randint(0,len(vertice_list)-1)
            while vertice_list[rnd] in v_neighs:
                rnd = random.randint(0,len(vertice_list)-1)
            v_neighs.append(vertice_list[rnd])
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

def count_min_probabilty(n):
    return math.log(n)/n

def count_prop_for_edges(num_of_edges, n):
    return (num_of_edges*2)/(n*(n-1))

def create_random_graph(n, p):
    edges = []
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
    return matrix


def dfs(matrix, start, visited):
    stack = [start]
    comp_v = []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            comp_v.append(vertex)
            stack.extend(get_neighbours(matrix,vertex) - visited)
    return comp_v

def get_neighbours(matrix,vertex):
    neighs = set()
    for index, col in enumerate(matrix[vertex]):
        if col == 1:
            neighs.add(index)
    return neighs

def get_components(matrix):
    components = []
    visited = set()
    for index, row in enumerate(matrix):
        if index not in visited:
            components.append(dfs(matrix, index, visited))
    return components

def prep_matrix(matrix):
    matrix = matrix.copy()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i == j: 
                    continue
                matrix[i][j] = sys.float_info.max
    return matrix

def floyd_algorithm(matrix):
    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(i+1):
                if matrix[k][j] != sys.float_info.max:
                    temp = matrix[i][k] + matrix[k][j]
                    if matrix[i][j] == sys.float_info.max or matrix[i][j] > temp:
                        matrix[i][j] = matrix[j][i] = temp
    return matrix

def count_avg_path(matrix, deep_copy=True):
    if deep_copy:
        sp_matrix = floyd_algorithm(prep_matrix(copy.deepcopy(matrix)))
    else:
        sp_matrix = floyd_algorithm(prep_matrix(matrix))
    total_len = sum([sum(row) for row in sp_matrix])
    n = len(sp_matrix)
    return (2*total_len)/(n*(n-1))

def count_avg_degree(matrix):
    #print(matrix)
    n = len(matrix)
    degrees = sum([sum(row) for row in matrix])
    return degrees/n

def get_component_matrix(base_matrix,vertices):
    n = len(vertices)
    c = 0
    matrix = [[0 for col in range(n)] for row in range(n)]
    mapping = {}
    for vertex in vertices:
        mapping[vertex] = c
        c+=1
    for vertex in vertices:
        for index, col in enumerate(base_matrix[vertex]):
            if index in vertices and col == 1:
                matrix[mapping[vertex]][mapping[index]] = matrix[mapping[index]][mapping[vertex]] = 1
    return matrix

def components_info(components):
    comp_count = len(components)
    components_sizes = [len(component) for component in components]
    #print(components_sizes)
    biggest_component = max(components_sizes)
    print("Number of components:{}\nBiggest component:{}".format(comp_count,biggest_component))
    return comp_count, biggest_component, components_sizes.index(biggest_component)

def remove_vertex(matrix, vertex):
    for index, col in enumerate(matrix[vertex]):
        if col == 1:
            matrix[vertex][index] = matrix[index][vertex] = 0
    return matrix

def get_vertex_with_highest_degree(matrix):
    degrees = [sum(row) for row in matrix]
    return degrees.index(max(degrees))

def simulate_attack(matrix, min_num_of_comp=1, graph_name = "default"):
    num_of_components = 1
    n = len(matrix)
    stats = []
    while num_of_components < min_num_of_comp:
        #print(num_of_components)
        #print(matrix)
        matrix = remove_vertex(matrix, get_vertex_with_highest_degree(matrix))
        num_of_components = analyse_network(matrix, stats)
    #draw_stats("attack",graph_name,stats)
    return stats

def simulate_random_failure(matrix, min_num_of_comp=1, graph_name="default"):
    removed_vertexes = []
    num_of_components = 1
    n = len(matrix)
    stats = []
    while num_of_components < min_num_of_comp:
        #print(matrix)
        rnd = random.randint(0,n-1)
        while rnd in removed_vertexes:
            rnd = random.randint(0,n-1)
        matrix = remove_vertex(matrix, rnd)
        removed_vertexes.append(rnd)
        num_of_components = analyse_network(matrix, stats)
    #draw_stats("failure",graph_name,stats)
    return stats

def analyse_network(matrix, stats):
    components = get_components(matrix)
    avg_degree = count_avg_degree(matrix)
    num_of_components, biggest_comp_size, biggest_comp_index = components_info(components)
    biggest_comp = get_component_matrix(matrix, components[biggest_comp_index])
    print("counting_avg")
    avg_path = count_avg_path(biggest_comp, False)
    print("done")
    print("Avg path: {}\nAvg degree: {}".format(avg_path,avg_degree))
    stats.append([biggest_comp_size,avg_path,avg_degree])
    return num_of_components

def draw_stats(graph_name,stats_failure, stats_attack):
    biggest_comp_f,avg_path_f,avg_degree_f = zip(*stats_failure)
    biggest_comp_a,avg_path_a,avg_degree_a = zip(*stats_attack)
    
    save_plot(biggest_comp_f, biggest_comp_a,graph_name, "Biggest component")
    save_plot(avg_path_f, avg_path_a,graph_name, "Average path")
    save_plot(avg_degree_f, avg_degree_a,graph_name, "Average degree")

def save_plot(datas_f, datas_a, graph_name, attribute_name):
    #print(datas)
    plt.plot(datas_f, label="Failure")
    plt.plot(datas_a, label="Attack")
    plt.title("{}-{}".format(graph_name,attribute_name))
    plt.legend()
    path = os.getcwd()+"/simulations/"+graph_name+attribute_name+".png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(path)
    plt.savefig(path)
    plt.clf()

def simulate_epidemic(graph_name, matrix, infection_init_prob, infection_prob, infection_dur, cycle_limit=False, cycle_count= 50):
    stable = []
    infected = []
    recovered = []
    nodes = {}
    for i in range(len(matrix)):
        rnd = random.random()
        infect= False
        if rnd < infection_init_prob:
            infect = True
        S = False if infect else True 
        I = True if infect else False
        R = False
        nodes[i] = {"S": S, "I": I, "R": R, "time_infected": 0}
    if cycle_limit:
        for i in range(cycle_count):
            nodes = one_epidemic_cycle(nodes,matrix,infection_prob, infection_dur)
            draw_epidemic_state(graph_name,nodes,matrix, i)
            
    else:
        cycle_num = 0
        while count_infected(nodes) > 0:
            nodes = one_epidemic_cycle(nodes,matrix,infection_prob, infection_dur)
            draw_epidemic_state(graph_name,nodes,matrix, cycle_num)
            cycle_num+=1
            

def count_infected(nodes):
    counter = 0
    for val in nodes.values():
        if val["I"]:
            counter+=1
    return counter

def one_epidemic_cycle(nodes, matrix, infection_prob, infection_dur):
    new_nodes = copy.deepcopy(nodes)
    for key, value in nodes.items():
        if value["I"]:
            neighbours = get_neighbours(matrix, key)
            for neighbour in neighbours:
                if nodes[neighbour]["R"] or nodes[neighbour]["I"]:
                    continue
                rnd = random.random()
                if rnd < infection_prob:
                    new_nodes[neighbour]["I"] = True
                    new_nodes[neighbour]["S"] = False
            t_i = value["time_infected"]
            if t_i == infection_dur:
                new_nodes[key]["I"] = False
                new_nodes[key]["R"] = True
            else:
                new_nodes[key]["time_infected"] += 1
    return new_nodes

def get_edges(matrix):
    edge_list = []
    for index, row in enumerate(matrix):
        for i in range(index+1):
            if matrix[index][i] == 1:
                edge_list.append([index,i])
    return edge_list

def draw_epidemic_state(graph_name, nodes, matrix, iteration):
    graph = nx.Graph()
    s_color = "#0000FF"
    i_color = "#FF0000"
    r_color = "#00FF00"
    node_colors = []
    for key,value in nodes.items():
        graph.add_node(key)
        if value["R"]:
            node_colors.append(r_color)
        elif value["I"]:
            node_colors.append(i_color)
        else:
            node_colors.append(s_color)
    edgs = get_edges(matrix)
    for edge in edgs:
        graph.add_edge(edge[0],edge[1])
    nx.draw(graph, node_color=node_colors, node_size=10)
    plt.savefig("{}{}.png".format(graph_name,iteration))
    plt.clf()

def count_num_of_edges(matrix):
    return sum([sum(row) for row in matrix])/2



#get_components(matrix)
#print(get_components(matrix))
#components_info(get_components(matrix))
#copy.deepcopy(matrix)
#simulate_random_failure(copy.deepcopy(matrix),450)
#print(len(edges))
#print(len(get_edges(matrix)))
#simulate_epidemic("plots/",matrix,0.2,0.2,2)
air_matrix = load_saved_graph(500)
baraba_matrix = create_barabasi_albert_graph_dynamic(500,3,10,[0,1,2,3,4,5,6,7,8,9])
num_of_edges = count_num_of_edges(baraba_matrix)
print(num_of_edges)
random_matrix = create_random_graph(500,count_prop_for_edges(num_of_edges,500)+0.0005)
num_of_edges = count_num_of_edges(random_matrix)
print(num_of_edges)
num_of_edges = count_num_of_edges(air_matrix)
print(num_of_edges)
names = ["Random","Air","Barabasi"]
matrixes = [random_matrix, air_matrix,baraba_matrix]
min_number_of_components = 100
for index, matrix in enumerate(matrixes):
    data_attack = simulate_attack(copy.deepcopy(matrix),graph_name=names[index], min_num_of_comp=min_number_of_components)
    data_failure = simulate_random_failure(copy.deepcopy(matrix),min_number_of_components,names[index])
    #simulate_epidemic("plots/{}".format(names[index]),matrix,0.2,0.2,2)
    draw_stats(names[index],data_failure,data_attack)
"""data_attack = simulate_attack(copy.deepcopy(matrix),graph_name="Zachari", min_num_of_comp=100)
data_failure = simulate_random_failure(copy.deepcopy(matrix),100,"Zachari")
draw_stats("Zachar",data_failure,data_attack)"""
"""print(count_avg_path(matrix,True))
print(count_avg_path(get_component_matrix(matrix, get_components(matrix)[0]),True))
print("***")
#simulate_attack(copy.deepcopy(matrix),450)
print("***********************")
#simulate_attack(copy.deepcopy(matrix),450)
print(count_avg_path(matrix))
print(count_avg_degree(matrix))"""