import csv
import random

import matplotlib.pyplot as plt


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


def link_selection_model(n, n0=3):
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
    return matrix


def link_selection_model_with_aging(n, n0=3):
    matrix = [[0 for i in range(n)] for j in range(n)]
    edges = []
    for i in range(n0 - 1):
        matrix[i][i + 1] = matrix[i + 1][i] = 1
        edges.append([i, i + 1])
    for i in range(n - n0):
        rnd = random.randint(i // 3, len(edges) - 1)
        rnd2 = random.random()
        older_vertex = min(edges[rnd])
        if rnd2 < 0.5:  # 1 / ((i + 1) / (older_vertex +1)):
            edges.append([edges[rnd][0], i + n0])
            matrix[edges[rnd][0]][i + n0] = matrix[i + n0][edges[rnd][0]] = 1
        else:
            edges.append([edges[rnd][1], i + n0])
            matrix[rnd][i + n0] = matrix[i + n0][rnd] = 1
    return matrix


def link_selection_model_with_deletion(n, r, n0=3):
    matrix = [[0 for i in range(n)] for j in range(n)]
    edges = []
    for i in range(n0 - 1):
        matrix[i][i + 1] = matrix[i + 1][i] = 1
        edges.append([i, i + 1])
    r_sum = 0
    n_of_vertex = n0
    for i in range(n - n0):
        rnd = random.randint(0, len(edges) - 1)
        rnd2 = random.random()
        if rnd2 < 0.5:
            edges.append([edges[rnd][0], i + n0])
            matrix[edges[rnd][0]][i + n0] = matrix[i + n0][edges[rnd][0]] = 1
        else:
            edges.append([edges[rnd][1], i + n0])
            matrix[rnd][i + n0] = matrix[i + n0][rnd] = 1
        r_sum += r
        n_of_vertex += 1
        while r_sum > 1:
            rnd = random.randint(0, n_of_vertex - 1)
            for index, col in enumerate(matrix[rnd]):
                if col == 1:
                    matrix[rnd][index] = matrix[index][rnd] = 0
            n_of_vertex -= 1
            r_sum -= 1
    return matrix


def get_neighbours(matrix, vertex):
    neighs = []
    for index, col in enumerate(matrix[vertex]):
        if col == 1:
            neighs.append(index)
    return neighs


def get_degree_centrality(matrix):
    return sorted([sum(row) for row in matrix])


def plot_data(data):
    degrees = {}
    data.reverse()
    for degree in data:
        degrees[degree] = degrees.get(degree, 0) + 1
    total = len(data)
    x = []
    y = []
    cumulation = 0
    degrees
    for key, value in degrees.items():
        x.append(key)
        cumulation += value / total
        y.append(cumulation)
    plt.plot(y, x)
    plt.show()


def copy_model(n, p, n0=3):
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
    return matrix


def count_age(t, ti, n0):
    age = t - (ti + n0)
    return 0 if age < 0 else age


def roulette(vertices, t, n0, v):
    probs = [count_age(t, vertice, n0) ** (-v) for vertice in vertices]
    total = sum(probs)
    if total == 0:
        print("shit")
    total = total if total > 0 else 1
    rnd = random.random()
    prob_sum = 0
    for index, prob in enumerate(probs):
        prob_sum += prob / total
        print("iterat")
        if prob_sum < rnd:
            print("*")
            return index
    print("*")
    return len(vertices) - 1


def copy_model_with_aging(n, p, n0=3):
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
            rnd = roulette(vertices, i, n0, -10)  # random.randint(0,len(vertices)-1)
            matrix[i + n0][vertices[rnd]] = matrix[vertices[rnd]][i + n0] = 1
    return matrix


def create_barabasi_albert_deletion(n, m, m0, r, vertice_list=None):
    """
    Updated version that allows to set vertice list instead of creating full graph. Also fixed varing num of edges
    """
    edges = []
    vertice_count = n - m0
    current_count = m0
    if vertice_list is None:
        vertice_list = [i for i in range(m0)]
        vertice_list.extend([i for i in range(1, m0 - 1)])
        vertice_list = sorted(vertice_list)
        """vertice_list = []
        for i in range(m0):
            for j in range(m0-1):
                vertice_list.append(i)"""
    neighbour_matrix = [[0 for x in range(n)] for y in range(n)]
    tmp1 = [i for i in range(m0)]
    tmp2 = [i for i in range(1, m0 - 1)]
    for index, item in enumerate(tmp2):
        neighbour_matrix[tmp1[index]][item] = neighbour_matrix[item][tmp1[index]] = 1
    neighbour_matrix[tmp1[m0 - 1]][tmp2[m0 - 3]] = neighbour_matrix[tmp2[m0 - 3]][tmp1[m0 - 1]] = 1
    # edges.append([i,j])
    """for i in range(m0):
        for j in range(m0):
            if i == j:
                continue
            edges.append([i,j])
            neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1"""
    # print(edges)
    # print(vertice_list)
    n_of_vertex = m0
    r_sum = 0
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
        r_sum += r
        removed_vertexes = []
        while r_sum > 1:
            rnd = random.randint(0, i + m0 - 1)
            while rnd in removed_vertexes:
                rnd = random.randint(0, m0 - 1)

            for index, col in enumerate(neighbour_matrix[rnd]):
                if col == 1:
                    neighbour_matrix[rnd][index] = neighbour_matrix[index][rnd] = 0
            while rnd in vertice_list:
                vertice_list.remove(rnd)
            removed_vertexes.append(rnd)
            r_sum -= 1
    # write_vertices_to_csv(edges,"barabasi-albert.csv")
    # print("Vertice list: {} {}".format(vertice_list, vertice_count))
    return neighbour_matrix


# safe_matrix_as_csv(link_selection_model(200,5),"link_model")
safe_matrix_as_csv(create_barabasi_albert_deletion(200, 3, 100, 2), "barabasi_with_deletion")
# safe_matrix_as_csv(link_selection_model_with_deletion(200,0.5,5),"link_model_deletion")
# safe_matrix_as_csv(link_selection_model_with_aging(200,5),"link_model_aging")
# safe_matrix_as_csv(copy_model_with_aging(200,0.2,5),"copy_model_with_aging")
# safe_matrix_as_csv(copy_model(200,0.2,5),"copy_model")
# plot_data(get_degree_centrality(link_selection_model(200,5)))
# plot_data(get_degree_centrality(copy_model(200,0.2,5)))
plot_data(get_degree_centrality(create_barabasi_albert_deletion(200, 3, 5, 0.5)))
plot_data(get_degree_centrality(create_barabasi_albert_deletion(200, 3, 5, 1)))
plot_data(get_degree_centrality(create_barabasi_albert_deletion(200, 3, 50, 3)))
