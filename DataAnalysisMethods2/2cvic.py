import csv
import math
import os
import random

import matplotlib.pyplot as plt


def count_min_probabilty(n):
    return math.log(n) / n


def count_prop_for_edges(num_of_edges, n):
    return (num_of_edges * 2) / (n * (n - 1))


def create_random_graph(n, p):
    edges = []
    # p by melo by vetsi nez 0,000921 pro 10000 vrcholu  - ostry prah nad kterym bude sit souvisla
    matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            col = j + i + 1
            if col >= n:
                break
            rnd = random.random()
            if rnd < p:
                matrix[i][col] = 1
                edges.append([i, col])
                matrix[col][i] = 1
    # [print(row) for row in matrix]
    # write_vertices_to_csv(edges,"random-chart.csv")

    return matrix


def create_barabasi_albert_graph_dynamic(n, m, m0):
    edges = []
    vertice_count = n - m0
    current_count = m0
    vertice_list = []
    for i in range(m0):
        for j in range(m0 - 1):
            vertice_list.append(i)
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(m0 - 1):
        for j in range(1, m0):
            if i == j:
                continue
            edges.append([i, j])
            neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1
    # print(edges)
    # print(vertice_list)
    for i in range(vertice_count):
        v_neighs = []
        for j in range(m):
            rnd = random.randint(0, len(vertice_list) - 1)
            if vertice_list[rnd] not in v_neighs:
                v_neighs.append(vertice_list[rnd])
        for neigh in v_neighs:
            vertice_list.insert(vertice_list.index(neigh), neigh)
            edges.append([current_count, neigh])
            neighbour_matrix[current_count][neigh] = neighbour_matrix[neigh][current_count] = 1
        for j in range(m):
            vertice_list.append(current_count)
        current_count += 1
    # write_vertices_to_csv(edges,"barabasi-albert.csv")
    # print("Vertice list: {} {}".format(vertice_list, vertice_count))
    return neighbour_matrix


def random_node_sampling(matrix, p, limit_on_p):
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
            if limit_on_p and len(nodes) > len(matrix) * p:
                print("Limiting with count {}".format(len(nodes)))
                return [edges, nodes]
    print(len(nodes))
    # print(len(edges))"""
    return [edges, nodes]


def degree_based_sampling(matrix, p, limit_on_p):
    n = len(matrix)
    edges = []
    nodes = []
    for i in range(n):
        rnd = random.random()
        degree = sum(matrix[i])
        passed = False
        if degree == 0:
            passed = p > rnd
        else:
            passed = p / degree  # p * degree < rnd#(p * degree) > rnd
        if passed:
            nodes.append(i)
            for index, item in enumerate(matrix[i]):
                if item > 0:
                    if index not in nodes:
                        nodes.append(index)
                    edges.append([i, index])
            if limit_on_p and len(nodes) > len(matrix) * p:
                print("Limiting with count {}".format(len(nodes)))
                return [edges, nodes]
    print(len(nodes))
    # print(len(edges))"""
    return [edges, nodes]


def cummulative_distribution(X, matrix, nodes, method_name, max_degree=0):
    degrees = [sum(row) for index, row in enumerate(matrix) if index in nodes]
    max_degree = max(degrees)
    rel_degrees = [degree / max_degree for degree in degrees]
    counts = {}
    for degree in sorted(degrees):
        counts[degree] = counts.get(degree, 0) + 1

    cumm = 0
    distribution = []
    for key in counts:
        counts[key] /= len(nodes)
        cumm += counts[key]
        distribution.append([key / max_degree, cumm])
    print("Dist len: {}".format(len(distribution)))

    # print(counts)
    # print(distribution)
    # print(distribution)
    plt.plot([point[0] for point in distribution], [point[1] for point in distribution], label=method_name)


def plot_chart(file_name):
    # plt.axis([0, 1.0, 0, 1.0])
    plt.title("Degree cummulative distribution")
    plt.xlabel("relative degree")
    plt.ylabel("relative cummulative frequency")
    plt.legend()
    path = os.getcwd() + file_name + ".png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # print(path)
    plt.savefig(path)
    plt.clf()


def get_max_degree(matrix, nodes):
    degrees = [sum(row) for index, row in enumerate(matrix) if index in nodes]
    return max(degrees)


def safe_matrix_as_csv(matrix, name):
    edges = []
    # print(sum([sum(row) for row in matrix]))
    for index, row in enumerate(matrix):
        for col_index, col in enumerate(row):
            if col == 1:
                if [index, col_index] in edges or [col_index, index] in edges:
                    # print("Yea")
                    continue
                edges.append([index, col_index])
    # print(len(edges))
    write_vertices_to_csv(edges, "{}.csv".format(name))


def write_vertices_to_csv(edges, name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)


node_count = 1000

ba_matrix = create_barabasi_albert_graph_dynamic(node_count, 2, 10)
rn_matrix = create_random_graph(node_count, count_prop_for_edges(2043, node_count))

prob = 0.15
limit_p = True
# nodes_list = random_node_sampling(ba_matrix, prob)[1]
# degree_based_sampling(ba_matrix, prob)

mx_degree = get_max_degree(ba_matrix, range(1000))
plt.clf()
cummulative_distribution(5, ba_matrix, range(1000), "base", mx_degree)
cummulative_distribution(5, ba_matrix, random_node_sampling(ba_matrix, prob, limit_p)[1], "RNA", mx_degree)
cummulative_distribution(5, ba_matrix, degree_based_sampling(ba_matrix, prob, limit_p)[1], "dbs", mx_degree)
plot_chart("\\barabasi-albert")
write_vertices_to_csv(random_node_sampling(ba_matrix, prob, limit_p)[0], "barabasi-rnd.csv")
write_vertices_to_csv(degree_based_sampling(ba_matrix, prob, limit_p)[0], "barabasi-dbs.csv")

cummulative_distribution(5, rn_matrix, range(1000), "base", mx_degree)
cummulative_distribution(5, rn_matrix, random_node_sampling(rn_matrix, prob, limit_p)[1], "RNA", mx_degree)
cummulative_distribution(5, rn_matrix, degree_based_sampling(rn_matrix, prob, limit_p)[1], "dbs", mx_degree)
plot_chart("\\random_chart")
write_vertices_to_csv(random_node_sampling(rn_matrix, prob, limit_p)[0], "random-rnd.csv")
write_vertices_to_csv(degree_based_sampling(rn_matrix, prob, limit_p)[0], "random-dbs.csv")

safe_matrix_as_csv(rn_matrix, "cvic2RandomGraph")
safe_matrix_as_csv(ba_matrix, "cvic2BarabasiGraph")
