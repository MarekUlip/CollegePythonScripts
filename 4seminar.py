import csv
import sys
"""import matplotlib.pyplot as plt
import numpy as np"""

# karate club
# 34 vrcholu

original_data = []

matrix = [[0 for col in range(34)] for row in range(34)]
with open('KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        row[0] = int(row[0])
        row[1] = int(row[1])
        original_data.append(row)
        col = int(row[1])-1
        row = int(row[0])-1
        matrix[row][col] = 1
        matrix[col][row] = 1

def prep_matrix(matrix):
            
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i == j: 
                    continue
                matrix[i][j] = sys.float_info.max
    return matrix

matrix = prep_matrix(matrix)
def floyd_algorithm(matrix):
    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
    return matrix

def count_closeness_centrality(matrix):
    closeness = []
    n = len(matrix)
    for i in range(n):
        closeness.append(0)
        for j in range(len(matrix[i])):
            closeness[i] = n/sum([x for x in matrix[i]])
    return closeness



matrix = floyd_algorithm(matrix)
closeness = count_closeness_centrality(matrix)   
#[print("{} {}".format(index, row)) for index, row in enumerate(matrix)]
#[print("{}. vertex closeness = {}".format(index+1,value)) for index, value in enumerate(closeness)]




from collections import defaultdict

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

def dijsktra(graph, initial, end, paths = [], shrtst = -1):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    suspicious = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    #print(suspicious)
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    if shrtst == -1:
        shrtst = len(path)
    if (len(path) > shrtst) or (path in paths):
        return
    paths.append(path)
    #for i in len(path):
    #print(path)
    for i in range(1, len(path)-1):
        edge = (path[i], path[i+1])
        if edge not in original_data:
            edge = (path[i+1], path[i])
            if edge not in original_data:
                continue
        #print(len(original_data))
        original_data.remove(edge)
        g = Graph()
        edgs = create_edges_list()
        for edg in edgs:
            g.add_edge(int(edg[0]), int(edg[1]), edg[2])
        dijsktra(g,initial,end,paths,shrtst)
        original_data.append(edge)
        #print(len(original_data))
        
    #path = path[1:len(path)-1]
    #[path.append(x) for x in suspicious]
    return path

"""def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path
"""
def create_edges_list():
    edges = []
    for i in range(len(original_data)):
        edges.append([original_data[i][0],original_data[i][1],1])
    return edges

edges = create_edges_list()
graph = Graph()
for edge in edges:
    graph.add_edge(int(edge[0]), int(edge[1]), edge[2])

org = []
for i in range(len(original_data)):
    org.append((original_data[i][0],original_data[i][1]))
original_data = org


"""pths = []
rpths = []
print(dijsktra(graph, 16, 17, pths))
shrtst = len(pths[0])
for pth in pths:
    if len(pth) == shrtst:
        pth = pth[1:len(pth)-1]
        if pth not in rpths:
            rpths.append(pth)
print(rpths)"""



#print(original_data)
points = []
for i in range(1,35):
    for j in range(1,35):
        if i == j or (j,i) in points:
            continue
        points.append((i,j))

dict_paths = []
for point in points:
    if matrix[point[0]-1][point[1]-1] > 1:
        pths = []
        rpths = []
        dijsktra(graph,point[0],point[1],pths)
        if(len(pths) == 0):
            rpths.append([])
            continue
        shrtst = len(pths[0])
        for pth in pths:
            if len(pth) == shrtst:
                pth = pth[1:len(pth)-1]
                if pth not in rpths:
                    rpths.append(pth)
        dict_paths.append(rpths)

def count_betweenes():
    res_dict = {}
    res_list = []
    for i in range(1,35):
        total = 0
        for shpths in dict_paths:
            suma = 0
            for item in shpths:
                for subitem in item:
                    if subitem == i:
                        suma+=1
                        break
            #print("{} {}".format(suma,len(shpths)))
            total += suma/len(shpths)
        res_dict[i] = total
        res_list.append(total)
    #print(res_dict)
    print(sorted(res_list))

count_betweenes()

#print(dict_paths)

