import csv
import random


def get_indexes_of_vertex(neighbour_matrix, vertex, exclude):
    indexes = []
    for index, col in enumerate(neighbour_matrix[vertex]):
        if col > 0 and index not in exclude:
            indexes.append(index)
    return indexes


def random_connection(neighbour_matrix, exclude, new_v_num):
    rnd = random.randint(0, new_v_num - 1)
    while rnd in exclude:
        print("Attempt to connect to already connected. Trying again.")
        rnd = random.randint(0, new_v_num - 1)
    neighbour_matrix[new_v_num][rnd] = neighbour_matrix[rnd][new_v_num] = 1
    exclude.append(rnd)


def get_neighbours(matrix, vertex):
    neighs = []
    for index, col in enumerate(matrix[vertex]):
        if col == 1:
            neighs.append(index)
    return neighs


def get_component_matrix(base_matrix, vertices):
    n = len(vertices)
    c = 0
    matrix = [[0 for col in range(n)] for row in range(n)]
    mapping = {}
    for vertex in vertices:
        mapping[vertex] = c
        c += 1
    for vertex in vertices:
        for index, col in enumerate(base_matrix[vertex]):
            if index in vertices and col == 1:
                matrix[mapping[vertex]][mapping[index]] = matrix[mapping[index]][mapping[vertex]] = 1
    return matrix


def count_average_clustering_coeficient_distribution(distribution):
    avg_res = {}
    for vertice in distribution:
        if vertice[1] not in avg_res:
            avg_res[vertice[1]] = [1, vertice[2]]
        else:
            avg_res[vertice[1]] = [avg_res[vertice[1]][0] + 1, avg_res[vertice[1]][1] + vertice[2]]

    for key, item in sorted(avg_res.items()):
        print("Degree {} has average clustering coeficient {}".format(key, item[1] / item[0]))


def get_edges(matrix):
    edge_list = []
    for index, row in enumerate(matrix):
        for i in range(index + 1):
            if matrix[index][i] == 1:
                edge_list.append([index, i])
    return edge_list


def write_vertices_to_csv(edges, name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)
