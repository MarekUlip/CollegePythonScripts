import csv
import copy
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from time import time
import random
import math

colors = ['red','green','blue','yellow','orange','purple','pink']

def load_data(path):
    points = []
    with open(path) as csv_file:
        readCSV = csv.reader(csv_file, delimiter=';')
        for row in readCSV:
            points.append([float(row[0]),float(row[1])])
    return points

def plot_data(data):
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1],c=get_colors(data[:,2]))
    plt.legend()
    plt.show()

def get_colors(cluster_indexes):
    for index in cluster_indexes:
        colors.append(colors[int(index)])
    return colors[:len(cluster_indexes)]
    
def euklid_distance(vec1, vec2):
    return np.sqrt( sum( [(vec1[i] - vec2[i])**2 for i in range(len(vec1))] ))

def get_min_from_row(row, threshold=0):
    min_n = 1000000000
    for i in row:
        if i < min_n and i > threshold:
            min_n = i
    return min_n
        
def get_two_closest_clusters(matrix):
    closest = [get_min_from_row(row) for row in matrix]
    max_val = min(closest)
    cluster_num = np.where(closest==max_val)[0][0]# closest.index(max_val)
    sec_cluster_num = np.where(matrix[cluster_num] == max_val)[0][0]#matrix[cluster_num].index(max_val)
    return [cluster_num, sec_cluster_num]

def linkage(matrix,linkage_params,clust_index_a,clust_index_b):
    #print("{}, {}".format(clust_index_a,clust_index_b))
    linkage_type = linkage_params[0]
    new_cluster_row = []
    if linkage_params[1] == 'single':
        new_cluster_row.append(100000000)
    elif linkage_params[1] == 'complete':
        new_cluster_row.append(-1)
    matrix = np.array(matrix)
    for index, row in enumerate(matrix):
        if index == clust_index_a or index == clust_index_b:
            continue
        new_cluster_row.append(linkage_type([row[clust_index_a],row[clust_index_b]]))#row[np.array([clust_index_a,clust_index_b])]))
    
    matrix = np.delete(matrix,[clust_index_a,clust_index_b],axis=0)
    matrix = np.delete(matrix,[clust_index_a,clust_index_b],axis=1)

    new_cluster_row = np.array(new_cluster_row)
    dummy_array = np.random.random(size=(len(new_cluster_row)-1))
    matrix = np.insert(matrix,0,dummy_array,axis=0)
    matrix = np.insert(matrix,0,new_cluster_row,axis=1)
    matrix[0] = new_cluster_row
    
    #np.hstack([new_cluster_row,matrix])
    #np.vstack([new_cluster_row,matrix])
    return matrix#matrix.tolist()
    
def remap_matrix_dict(index_a,index_b,old_map:dict):
    new_map = {0:None}
    modificator = 1
    joined_cluster = []
    for key,value in old_map.items():
        if key == index_a or key == index_b:
            modificator-=1
            joined_cluster.extend(value)
            continue
        new_map[key+modificator] = value
    new_map[0] = joined_cluster

    return new_map#dict(OrderedDict(sorted(new_map.items(), key = lambda t: t[0])))


def create_dist_matrix(data,linkage_type):
    matrix = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                if linkage_type == 'single':
                    matrix[i,j] = 100000000
                elif linkage_type == 'complete':
                    matrix[i,j] = -1
                continue
            matrix[i,j] = euklid_distance(data[i],data[j])
    return matrix#matrix.tolist()

def aglomerative_clustering(data, matrix,linkage_type, cut_at_len=3):
    clusters_list = []
    new_cluster=[[i] for i in range(len(data))]
    clusters_list.append(new_cluster)
    cluster_mapping = {}
    for i in range(len(data)):
        cluster_mapping[i] = [i]
    
    while len(new_cluster)>1:
        new_cluster.clear()
        index_a, index_b = get_two_closest_clusters(matrix)
        #print(len(matrix))
        matrix = linkage(matrix,linkage_type,index_a,index_b)
        cluster_mapping = remap_matrix_dict(index_a,index_b,cluster_mapping)
        for value in cluster_mapping.values():
            new_cluster.append(value)
        clusters_list.append(copy.deepcopy(new_cluster))
        print(len(new_cluster))
        #input()
    print(len(clusters_list[-cut_at_len]))
    return clusters_list[-cut_at_len]


def output_cluster(cluster, num_of_clusters, data):
    #print(cluster)
    #mode="slice" name="" timerepresentation="timestamp" timestamp=
    cluster_less_color = "#FF0000"
    colors = ["#8B008B", "#A9A9A9", "#1E90FF", "#F08080", "#7B68EE", "#2F4F4F", "#A0522D", "#483D8B", "#48D1CC", "#00FF00", "#B8860B" ]
    node_colors = []
    points = []
    for sub_clust in cluster:
        if type(sub_clust) is list:
            #print("new clusteeer")
            color = colors.pop()
            for item in sub_clust:
                node_colors.append(color)
                points.append(data[item])
                #to_output.append("{},\"{}\",{},{}\n".format(item+1,item+1,color,num_of_clusters))
        else:
            node_colors.append(cluster_less_color)
    node_colors = node_colors[:len(points)]
    points = np.array(points)
    plt.scatter(points[:,0],points[:,1],c=node_colors)
    plt.show()    

def euklid_distance_points(point_a, point_b):
        if type(point_a) is not list:
            point_a = [point_a]
        if type(point_b) is not list:
            point_b = [point_b]
        if len(point_a) != len(point_b):
            print("Lengths of points do not match. Cannot count distance. Returning")
            return 
        return math.sqrt(sum([(x-y)**2 for x,y in zip(point_a,point_b)]))

def k_means(data, k, minimum_cluster_change):
    #TODO clear clusters after each iteration, make centroid creatin n dimensional
    centroids = []
    for i in range(k):
        centroids.append(data[random.randint(0,len(data))])
    clusters, cluster_items_indexes = create_new_clusters(data, centroids, k)
    output_cluster(cluster_items_indexes,k,data)
    prev_centroids = centroids.copy()
    centroids = find_new_centroids(clusters,centroids)
    centroid_change = count_centroids_difference(centroids,prev_centroids)

    while centroid_change > minimum_cluster_change:
        clusters, cluster_items_indexes = create_new_clusters(data, centroids, k)
        output_cluster(cluster_items_indexes,k,data)
        prev_centroids = centroids.copy()
        centroids = find_new_centroids(clusters,centroids)
        centroid_change = count_centroids_difference(centroids,prev_centroids)
    return cluster_items_indexes

def find_new_centroids( clusters, centroids):
    dim = len(centroids[0])
    for index, cluster in enumerate(clusters):
        if len(cluster) == 0:
            centroids[index] = [random.randint(0,10) for x in range(dim)]
            continue
        centroid = [0 for x in range(dim)]
        for point in cluster:
            for index2, item in enumerate(point):
                centroid[index2]+=item
        centroid = [item/len(cluster) for item in centroid]
        centroids[index] = centroid
    return centroids

def count_centroids_difference( centroids, prev_centroids):
    centroid_change = 0
    for index, centroid in enumerate(centroids):
        centroid_change += euklid_distance_points(centroid,prev_centroids[index])
    return centroid_change

def create_new_clusters( data, centroids, k):
    clusters = [[] for x in range(k)]
    cluster_items_indexes = [[] for x in range(k)]
    for index, item in enumerate(data):
        if item not in centroids:
            closest_centroid = [0,99999999999]
            for index2, centroid in enumerate(centroids):
                #clusters[random.randint(0,k)].append(item)
                distance = euklid_distance_points(item,centroid)
                if distance < closest_centroid[1]:
                    closest_centroid[0] = index2
                    closest_centroid[1] = distance
            clusters[closest_centroid[0]].append(item)
            cluster_items_indexes[closest_centroid[0]].append(index)
    return clusters, cluster_items_indexes

data = load_data("clusters5n.csv")
options = [[min,'single'],[max,'complete']]
option = options[1]
dist_matrix = create_dist_matrix(data,option[1])
num_of_clusters = 3
output_cluster(aglomerative_clustering(data,dist_matrix,option,num_of_clusters),num_of_clusters,data)
#k_means(data,num_of_clusters,0.00001)