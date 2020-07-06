import copy
import csv
import math

import matplotlib.pyplot as plt
import networkx as nx

matrix = [[0 for col in range(34)] for row in range(34)]
nodes = []
edges = []

with open('1KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        edges.append(row.copy())
        row[0] = int(row[0])
        row[1] = int(row[1])
        # original_data.append(row)
        col = int(row[1]) - 1
        row = int(row[0]) - 1
        matrix[row][col] = 1
        matrix[col][row] = 1
        nodes.append(row)


def count_similarity(matrix):
    n = len(matrix)
    sim_matrix = [[0 for col in range(n)] for row in range(n)]
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
            sim_matrix[i][j] = len(shared_nodes) / math.sqrt(len(a_nodes) * len(b_nodes))
    return sim_matrix


def find_index_in_sub_cluster(cluster, a, b=None):
    a_cluster_index = b_cluster_index = -1
    if b is None:
        b_cluster_index = -2
    for index, c in enumerate(cluster):
        if type(c) is list:
            if a in c:
                a_cluster_index = index
                continue
            if b in c:  # Maybe check for none
                b_cluster_index = index
        if a_cluster_index != -1 and b_cluster_index != -1:
            break
    return [a_cluster_index, b_cluster_index]


def clustering(matrix):
    clustered_nodes = []
    clusters = []
    cluster = [i for i in range(len(matrix))]
    cluster_count = len(matrix)
    while cluster_count > 1:
        print(cluster_count)
        a, b = simple_linkage(matrix)
        print("a: {}, b: {}".format(a, b))
        if a in clustered_nodes and b in clustered_nodes:
            """a_cluster_index = b_cluster_index = -1
            for index, c in enumerate(cluster):
                if type(c) is list:
                    if a in c:
                        a_cluster_index = index
                        continue
                    if b in c:
                        b_cluster_index = index
                if a_cluster_index != -1 and b_cluster_index != -1:
                    break"""
            a_cluster_index, b_cluster_index = find_index_in_sub_cluster(cluster, a, b)
            cluster[a_cluster_index].extend(cluster[b_cluster_index])
            update_matrix(cluster[a_cluster_index], matrix)
            cluster.remove(cluster[b_cluster_index])
        elif a in clustered_nodes:
            a_cluster_index = find_index_in_sub_cluster(cluster, a)[0]
            cluster[a_cluster_index].append(b)
            update_matrix(cluster[a_cluster_index], matrix)
            cluster.remove(b)
            clustered_nodes.append(b)
        elif b in clustered_nodes:
            b_cluster_index = find_index_in_sub_cluster(cluster, b)[0]
            cluster[b_cluster_index].append(a)
            update_matrix(cluster[b_cluster_index], matrix)
            cluster.remove(a)
            clustered_nodes.append(a)
        else:
            new_cluster_index = cluster.index(a)
            cluster[new_cluster_index] = [a, b]
            update_matrix(cluster[new_cluster_index], matrix)
            # cluster.remove(a)
            cluster.remove(b)
            clustered_nodes.append(a)
            clustered_nodes.append(b)
        clusters.append(copy.deepcopy(cluster))
        cluster_count -= 1
    print(len(clusters))
    return clusters


def update_matrix(community, matrix):
    for i in community:
        for j in community:
            matrix[i][j] = 0


def simple_linkage(matrix):
    closest = [max(row) for row in matrix]
    max_val = max(closest)
    vertex_num = closest.index(max_val)
    sec_vertex_num = matrix[vertex_num].index(max_val)
    return [vertex_num, sec_vertex_num]


def complete_linkage(matrix):
    closest = [min(row) for row in matrix]
    max_val = min(closest)
    vertex_num = closest.index(max_val)
    sec_vertex_num = matrix[vertex_num].index(max_val)
    return [vertex_num, sec_vertex_num]


def output_cluster_no_timeline(cluster, num_of_clusters):
    to_output = ["nodedef>name VARCHAR, label VARCHAR, color VARCHAR\n"]
    edge_part = "edgedef> node1,node2\n"
    cluster_less_color = "#FF0000"
    colors = ["#8B008B", "#A9A9A9", "#1E90FF", "#F08080", "#7B68EE", "#2F4F4F", "#A0522D", "#483D8B", "#48D1CC",
              "#00FF00", "#B8860B", ]
    for sub_clust in cluster:
        if type(sub_clust) is list:
            color = colors.pop()
            for item in sub_clust:
                to_output.append("{},\"{}\",{}\n".format(item + 1, item + 1, color))
        else:
            to_output.append("{},\"{}\",{}\n".format(sub_clust + 1, sub_clust + 1, cluster_less_color))
    to_output.append(edge_part)
    for edge in edges:
        to_output.append("{},{}\n".format(edge[0], edge[1]))
    with open("clusters-{}.gdf".format(num_of_clusters), "w") as f:
        for item in to_output:
            f.write(item)


def output_cluster_gdf(cluster, num_of_clusters):
    to_output = ["nodedef>name VARCHAR, label VARCHAR, color VARCHAR, timeline_dynamic VARCHAR\n"]
    edge_part = "edgedef> node1,node2\n"
    cluster_less_color = "#FF0000"
    colors = ["#8B008B", "#A9A9A9", "#1E90FF", "#F08080", "#7B68EE", "#2F4F4F", "#A0522D", "#483D8B", "#48D1CC",
              "#00FF00", "#B8860B", ]
    for sub_clust in cluster:
        if type(sub_clust) is list:
            color = colors.pop()
            for item in sub_clust:
                to_output.append("{},\"{}\",{},{}\n".format(item + 1, item + 1, color, num_of_clusters))
        else:
            to_output.append(
                "{},\"{}\",{},{}\n".format(sub_clust + 1, sub_clust + 1, cluster_less_color, num_of_clusters))
    to_output.append(edge_part)
    for edge in edges:
        to_output.append("{},{}\n".format(edge[0], edge[1]))
    with open("clusters-{}.gdf".format(num_of_clusters), "w") as f:
        for item in to_output:
            f.write(item)


def output_cluster(cluster, num_of_clusters):
    graph = nx.Graph(name=str(num_of_clusters), timerepresentation="timestamp", timestamp=str(num_of_clusters))
    print(graph.graph)
    print("f")
    # mode="slice" name="" timerepresentation="timestamp" timestamp=
    cluster_less_color = "#FF0000"
    colors = ["#8B008B", "#A9A9A9", "#1E90FF", "#F08080", "#7B68EE", "#2F4F4F", "#A0522D", "#483D8B", "#48D1CC",
              "#00FF00", "#B8860B", ]
    node_colors = []
    for sub_clust in cluster:
        if type(sub_clust) is list:
            color = colors.pop()
            for item in sub_clust:
                graph.add_node(item + 1)
                node_colors.append(color)
                graph.node[item + 1]['viz'] = {'color': {'r': color, 'g': color, 'b': color, 'a': 0}}
                # to_output.append("{},\"{}\",{},{}\n".format(item+1,item+1,color,num_of_clusters))
        else:
            graph.add_node(sub_clust + 1)
            node_colors.append(cluster_less_color)
            graph.node[sub_clust + 1]['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
    for edge in edges:
        graph.add_edge(int(edge[0]), int(edge[1]))
        # to_output.append("{},{}\n".format(edge[0],edge[1]))
    nx.write_gexf(graph, "clusters-{}.gexf".format(num_of_clusters))
    nx.draw(graph, node_color=node_colors)
    plt.show()


similarity_matrix = count_similarity(matrix)
similarity_matrix_work = similarity_matrix.copy()
clusters = clustering(similarity_matrix_work)
[print("{}: {}".format(index, row)) for index, row in enumerate(clusters)]
clust_num = 17
output_cluster(clusters[clust_num], clust_num + 1)
# simple_linkage(similarity_matrix)
# [print(row) for row in count_similarity(matrix)]
