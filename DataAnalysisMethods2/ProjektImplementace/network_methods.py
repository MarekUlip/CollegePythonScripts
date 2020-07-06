import copy
import csv
import random

import matplotlib.pyplot as plt
import networkx as nx
from helper_network_methods import *


class NetworkMethods:
    def __init__(self, output_path=""):
        self.matrix = None
        self.size = 0
        self.output_path = output_path
        self.analysis_summary = []

    def load_network_from_csv(self, path, first_node_index, delimeter=","):
        with open(path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=delimeter)
            edges = []
            max_node_num = 0
            for row in readCSV:
                edges.append(row)
                row[0] = int(row[0]) - first_node_index
                if row[0] > max_node_num:
                    max_node_num = row[0]

                row[1] = int(row[1]) - first_node_index
                if row[1] > max_node_num:
                    max_node_num = row[1]
                edges.append([row[0], row[1]])
                # original_data.append(row)
            max_node_num += 1
            matrix = [[0 for x in range(max_node_num)] for y in range(max_node_num)]
            print(max_node_num)
            for edge in edges:
                col = int(edge[1])
                row = int(edge[0])
                matrix[row][col] = 1
                matrix[col][row] = 1

            self.size = max_node_num
            self.matrix = matrix

    def save_summary(self):
        with open("{}/{}".format(self.output_path, "summary.txt"), "a+") as f:
            for line in self.analysis_summary:
                f.write("{}\n".format(line))
        self.analysis_summary.clear()

    def generate_barabasi_albert_graph(self, n, m, n0, base_is_full=True):
        """
        Updated version that allows to set vertice list instead of creating full graph. Also fixed varing num of edges
        """
        edges = []
        vertice_count = n - n0
        current_count = n0
        vertice_list = []
        if base_is_full:
            for i in range(n0):
                for j in range(n0 - 1):
                    vertice_list.append(i)
        else:
            # generates connected network with all nodes connected in "straight line" i.e. for n0 = 3 [1,2,2,3]
            vertice_list = [i for i in range(n0)]
            vertice_list.extend([i for i in range(1, n0 - 1)])
            vertice_list = sorted(vertice_list)
        neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
        for i in range(n0):
            for j in range(n0):
                if i == j:
                    continue
                edges.append([i, j])
                neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1
        for i in range(vertice_count):
            v_neighs = []
            for j in range(m):
                rnd = random.randint(0, len(vertice_list) - 1)
                while vertice_list[rnd] in v_neighs:
                    rnd = random.randint(0, len(vertice_list) - 1)
                v_neighs.append(vertice_list[rnd])
            for neigh in v_neighs:
                vertice_list.insert(vertice_list.index(neigh), neigh)
                edges.append([current_count, neigh])
                neighbour_matrix[current_count][neigh] = neighbour_matrix[neigh][current_count] = 1
            for j in range(m):
                vertice_list.append(current_count)
            current_count += 1
        self.size = n
        self.matrix = neighbour_matrix

    def generate_random_graph(self, n, p):
        # p by melo by vetsi nez 0,000921 pro 10000 vrcholu  - ostry prah nad kterym bude sit souvisla
        matrix = [[0 for x in range(n)] for y in range(n)]
        for i in range(n):
            for j in range(n):
                col = j + i + 1
                if col >= n:
                    break
                rnd = random.random()
                if rnd < p:
                    matrix[i][col] = matrix[col][i] = 1
        self.size = n
        self.matrix = matrix

    def copy_model(self, n, p, n0=3):
        matrix = [[0 for i in range(n)] for j in range(n)]
        for i in range(n0 - 1):
            matrix[i][i + 1] = matrix[i + 1][i] = 1
        for i in range(n - n0):
            rnd = random.randint(0, i + n0 - 1)
            rnd2 = random.random()
            if rnd2 < p:
                matrix[i + n0][rnd] = matrix[rnd][i + n0] = 1
            else:
                vertices = get_neighbours(matrix, rnd)
                rnd = random.randint(0, len(vertices) - 1)
                matrix[i + n0][vertices[rnd]] = matrix[vertices[rnd]][i + n0] = 1
        self.size = n
        self.matrix = matrix

    def link_selection_model(self, n, n0=3):
        matrix = [[0 for i in range(n)] for j in range(n)]
        edges = []
        for i in range(n0 - 1):
            matrix[i][i + 1] = matrix[i + 1][i] = 1
            edges.append([i, i + 1])
        for i in range(n - n0):
            rnd = random.randint(0, len(edges) - 1)
            rnd2 = random.random()
            if rnd2 < 0.5:
                edges.append([edges[rnd][0], i + n0])
                matrix[edges[rnd][0]][i + n0] = matrix[i + n0][edges[rnd][0]] = 1
            else:
                edges.append([edges[rnd][1], i + n0])
                matrix[rnd][i + n0] = matrix[i + n0][rnd] = 1
        self.size = n
        self.matrix = matrix

    def biancony_model(self, p, m, m0, n):
        vertice_count = 2
        neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
        neighbour_matrix[0][1] = neighbour_matrix[1][0] = 1
        for i in range(m0 - 2):
            rnd = random.randint(0, vertice_count - 1)
            neighbour_matrix[vertice_count][rnd] = neighbour_matrix[rnd][vertice_count] = 1
            vertice_count += 1
        for i in range(n - m0):
            rnd = random.randint(0, vertice_count - 1)
            print("rnd: {}, v_c: {}".format(rnd, vertice_count))
            neighbour_matrix[vertice_count][rnd] = neighbour_matrix[rnd][vertice_count] = 1
            exclude = [vertice_count]
            for j in range(m - 1):
                chance = random.random()
                if chance < p:
                    available_vertexes = get_indexes_of_vertex(neighbour_matrix, rnd, exclude)
                    if len(available_vertexes) == 1:
                        rnd_vertex = 0
                    elif len(available_vertexes) == 0:
                        random_connection(neighbour_matrix, exclude, vertice_count)
                        continue
                    else:
                        rnd_vertex = random.randint(0, len(available_vertexes) - 1)
                    exclude.append(available_vertexes[rnd_vertex])
                    neighbour_matrix[vertice_count][available_vertexes[rnd_vertex]] = \
                        neighbour_matrix[available_vertexes[rnd_vertex]][vertice_count] = 1
                else:
                    random_connection(neighbour_matrix, exclude, vertice_count)
            vertice_count += 1
        self.size = n
        self.matrix = neighbour_matrix

    def analyse_network(self, network=None, overwrite=False, params=None):
        matrix_bckup = []
        if network is None and self.matrix is None:
            print("No network to analyse.")
            return
        if network is None:
            network = copy.deepcopy(self.matrix)
        elif overwrite:
            self.matrix = network
            self.size = len(network)

        if network is not None and not overwrite:
            matrix_bckup = copy.deepcopy(self.matrix)
            self.matrix = network
        self.analysis_summary.append("")
        if params is not None:
            self.analysis_summary.extend(params)
        size = len(network) if network is not None else self.size
        self.analysis_summary.append("Number of nodes: {}".format(size))
        self.analysis_summary.append("Number of edges: {}".format(self.count_num_of_edges()))
        self.analysis_summary.append("Average degree: {}".format(self.count_avg_degree()))
        min_d, max_d = self.get_max_degree()
        self.analysis_summary.append("Max degree: {}".format(max_d))
        self.analysis_summary.append("Min degree: {}".format(min_d))
        # self.analysis_summary.append("Clustering coeficient: {}".format((self.clustering_koeficients())))
        if network is not None and not overwrite:
            self.matrix = matrix_bckup
        self.save_summary()

    def safe_matrix_as_csv(self, name):
        self.analysis_summary.append("\n")
        self.analysis_summary.append("Graph path: {}".format(name))
        edges = []
        # print(sum([sum(row) for row in matrix]))
        for index, row in enumerate(self.matrix):
            for col_index, col in enumerate(row):
                if col == 1:
                    if [index, col_index] in edges or [col_index, index] in edges:
                        # print("Yea")
                        continue
                    edges.append([index, col_index])
        # print(len(edges))
        write_vertices_to_csv(edges, "{}.csv".format(name))

    def count_num_of_edges(self):
        return sum([sum(row) for row in self.matrix]) / 2

    def count_avg_degree(self):
        # print(matrix)
        n = len(self.matrix)
        degrees = sum([sum(row) for row in self.matrix])
        if n == 0:
            return 0
        return degrees / n

    def get_max_degree(self, nodes=None):
        if len(self.matrix) == 0:
            return 0, 0
        if nodes is None:
            nodes = [i for i in range(self.size)]
        degrees = [sum(row) for index, row in enumerate(self.matrix) if index in nodes]
        return min(degrees), max(degrees)

    def count_average_clustering_coeficient(self, n=None):
        matrix = self.matrix
        if n is None:
            n = len(matrix[0])
        clusters = []
        distribution = []
        for i in range(1, n):
            res = self.clustering_koeficient(i)
            if res == -1:
                continue
            clusters.append(res)
            distribution.append([i, sum([x for x in matrix[i - 1]]), res])
            # print("{}. {}".format(i,clustering_koeficient(i)))
        return clusters, distribution

    def clustering_koeficient(self, vertice=None):
        matrix = self.matrix
        if vertice is None:
            vertice = [i for i in range(len(matrix[0]))]
        vertice_neigh = []
        for index, item in enumerate(matrix[vertice - 1]):
            if item == 1:
                vertice_neigh.append(index + 1)
        neighbours = sum([x for x in matrix[vertice - 1]])
        vertex = []
        for vert in vertice_neigh:
            for vert2 in vertice_neigh:
                if vert == vert2 or ((vert2, vert) in vertex):
                    continue
                vertex.append((vert, vert2))
        num_of_edges = 0
        for v in vertex:
            if matrix[v[0] - 1][v[1] - 1] == 1:
                num_of_edges += 1
        if neighbours * (neighbours - 1) == 0:
            return -1
        clstr_koef = (2 * num_of_edges) / (neighbours * (neighbours - 1))
        return clstr_koef

    def clustering_koeficients(self):
        matrix = self.matrix
        clstr_koefs = []
        vertices = [i for i in range(len(matrix[0]))]
        for vertice in vertices:
            vertice_neigh = []
            for index, item in enumerate(matrix[vertice - 1]):
                if item == 1:
                    vertice_neigh.append(index + 1)
            neighbours = sum([x for x in matrix[vertice - 1]])
            vertex = []
            for vert in vertice_neigh:
                for vert2 in vertice_neigh:
                    if vert == vert2 or ((vert2, vert) in vertex):
                        continue
                    vertex.append((vert, vert2))
            num_of_edges = 0
            for v in vertex:
                if matrix[v[0] - 1][v[1] - 1] == 1:
                    num_of_edges += 1
            if neighbours * (neighbours - 1) == 0:
                return -1
            clstr_koefs.append((2 * num_of_edges) / (neighbours * (neighbours - 1)))
        return clstr_koefs

    def k_core(self, k):
        """
        Finds core of nodes that have degree k (after removal of previous cores)
        :param k:
        :return: list of nodes nodes that belongs to k class
        """
        matrix = copy.deepcopy(self.matrix)
        node_degrees = [sum(row) for row in matrix]

        k_classes = []
        removed = 999
        while removed > 0:
            removed = 0
            for vertex, node_degree in enumerate(node_degrees):
                if node_degree < k:
                    for index, col in enumerate(matrix[vertex]):
                        if col == 1:
                            removed += 1
                            matrix[vertex][index] = matrix[index][vertex] = 0
            node_degrees = [sum(row) for row in matrix]
        for index, degree in enumerate(node_degrees):
            if degree > 0:
                k_classes.append(index)
        return k_classes

    def random_node_sampling(self, p):
        """
        Creates network sample
        :param p: size of sample in %
        :return: list of lists in form of [edges, nodes]
        """
        p *= 0.01
        matrix = self.matrix
        n = len(matrix)
        edges = []
        nodes = []
        for i in range(n):
            rnd = random.random()
            if p > rnd:
                nodes.append(i)
                for index, item in enumerate(matrix[i]):
                    if item > 0:
                        if index not in nodes:
                            nodes.append(index)
                        edges.append([i, index])
                if len(nodes) > n * p:
                    print("Limiting with count {}".format(len(nodes)))
                    return [edges, nodes]
        print(len(nodes))
        # print(len(edges))"""
        return [edges, nodes]

    def simulate_epidemic(self, graph_name, infection_init_prob, infection_prob, infection_dur, cycle_limit=False,
                          cycle_count=50):
        matrix = self.matrix
        infection_prob /= 100
        infection_init_prob /= 100
        nodes = {}
        for i in range(len(matrix)):
            rnd = random.random()
            infect = False
            if rnd < infection_init_prob:
                infect = True
            S = False if infect else True
            I = True if infect else False
            R = False
            nodes[i] = {"S": S, "I": I, "R": R, "time_infected": 0}
        if cycle_limit:
            for i in range(cycle_count):
                nodes = self.one_epidemic_cycle(nodes, infection_prob, infection_dur)
                self.draw_epidemic_state(graph_name, nodes, i)

        else:
            cycle_num = 0
            infected_count = self.count_infected(nodes)
            while infected_count > 0:
                if infection_dur < 0 and infected_count == self.size:
                    break
                nodes = self.one_epidemic_cycle(nodes, infection_prob, infection_dur)
                self.draw_epidemic_state(graph_name, nodes, cycle_num)
                cycle_num += 1

    def count_infected(self, nodes):
        counter = 0
        for val in nodes.values():
            if val["I"]:
                counter += 1
        return counter

    def one_epidemic_cycle(self, nodes, infection_prob, infection_dur):
        matrix = self.matrix
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
                if infection_dur > 0 and t_i == infection_dur:
                    new_nodes[key]["I"] = False
                    new_nodes[key]["R"] = True
                else:
                    new_nodes[key]["time_infected"] += 1
        return new_nodes

    def draw_epidemic_state(self, graph_name, nodes, iteration):
        matrix = self.matrix
        graph = nx.Graph()
        s_color = "#0000FF"
        i_color = "#FF0000"
        r_color = "#00FF00"
        i = []
        s = []
        r = []
        node_colors = []
        for key, value in nodes.items():
            graph.add_node(key)
            if value["R"]:
                node_colors.append(r_color)
                r.append(key)
            elif value["I"]:
                node_colors.append(i_color)
                i.append(key)
            else:
                node_colors.append(s_color)
                s.append(key)
        edgs = get_edges(matrix)
        for edge in edgs:
            graph.add_edge(edge[0], edge[1])
        nx.draw(graph, node_color=node_colors, node_size=10)
        plt.savefig("{}{}.png".format(graph_name, iteration))
        plt.clf()
        self.analysis_summary.append("Epidemic state after {}. iteration".format(iteration + 1))
        self.analysis_summary.append("S {}".format(s))
        self.analysis_summary.append("I {}".format(i))
        self.analysis_summary.append("R {}".format(r))
