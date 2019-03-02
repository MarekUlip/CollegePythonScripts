import math
import csv
import os
import random
from operator import itemgetter
import numpy as np
dataset = []
dataset_dict = {}

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

with open('1testDataIris.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    petal_name = 4
    for row in readCSV:
        """if row[petal_name] not in dataset_dict:
            dataset_dict[row[petal_name]] = []"""
        r = []
        for item in row:
            if isfloat(item):
                r.append(float(item))
        #dataset_dict[row[petal_name]].append(r)
        dataset.append(r)

def euklid_distance_points(point_a, point_b):
        if type(point_a) is not list:
            point_a = [point_a]
        if type(point_b) is not list:
            point_b = [point_b]
        if len(point_a) != len(point_b):
            print("Lengths of points do not match. Cannot count distance. Returning")
            return 
        return sum([(x-y)**2 for x,y in zip(point_a,point_b)])

def euklid_distance_points_real(point_a, point_b):
        if type(point_a) is not list:
            point_a = [point_a]
        if type(point_b) is not list:
            point_b = [point_b]
        if len(point_a) != len(point_b):
            print("Lengths of points do not match. Cannot count distance. Returning")
            return 
        return math.sqrt(sum([(x-y)**2 for x,y in zip(point_a,point_b)]))

def create_similarity_matrix(datas):
    data_size = len(datas)
    points = [[0 for j in range(data_size)] for i in range(data_size)]
    for i in range(data_size):
        for j in range(data_size):
            if i == j:
                continue
            points[i][j] = gaussian_kernel(datas[i], datas[j])#euklid_distance_points(datas[i], datas[j])
    return points

"""def count_e_radius_euklid(datas, e):
    data_size = len(datas)
    points = [[0 for j in range(data_size)] for i in range(data_size)]
    points_list = []
    for i in range(data_size):
        for j in range(data_size):
            if i == j:
                continue
            if euklid_distance_points_real(datas[i], datas[j]) < e:
                points[i][j] = 1
                points_list.append([i,j])
    return points_list"""

def count_e_radius(distance_matrix, e):
    data_size = len(distance_matrix)
    points = [[0 for j in range(data_size)] for i in range(data_size)]
    #points_list = []
    for i in range(data_size):
        for j in range(data_size):
            if i == j:
                continue
            if distance_matrix[i][j] >= e:
                points[i][j] = distance_matrix[i][j]#gaussian_kernel(dataset[i], dataset[j])           
    #print("e radius edge count {}".format(len(points_list)))
    return points

def knn(distance_matrix, k):
    n = len(distance_matrix)
    points = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for index in np.argsort(distance_matrix[i])[-k:]:
            points[i][index] = distance_matrix[i][index]#gaussian_kernel(dataset[i], dataset[index])
    #print("knn edge count {}".format(len(points_list)))
    return points

def knn_e_combined(distance_matrix, e, k):
    data_size = len(distance_matrix)
    points = [[0 for j in range(data_size)] for i in range(data_size)]
    #points_list = []
    for i in range(data_size):
        neighs_found = 0
        for j in range(data_size):
            if i == j:
                continue
            if distance_matrix[i][j] >= e:
                points[i][j] = distance_matrix[i][j]
                neighs_found+=1
        if neighs_found < k:
            print("deploy knn")
            for index in np.argsort(distance_matrix[i])[-k:]:
                points[i][index] = distance_matrix[i][index]
    return points



def write_vertices_to_csv(edges,name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)
            

def gaussian_kernel(point_a, point_b):
    return math.exp(-(euklid_distance_points(point_a,point_b)/2))

def convert_to_point_list(matrix):
    edges = []
    for index, row in enumerate(matrix):
        for jndex, col in enumerate(row):
            if col > 0:
                edges.append([index, jndex])
    return edges

similarity_mat = create_similarity_matrix(dataset)
write_vertices_to_csv(convert_to_point_list(count_e_radius(similarity_mat,0.75)), "e-radius.csv") #1.5
write_vertices_to_csv(convert_to_point_list(knn(similarity_mat,5)), "knn.csv") #k 5 sada rozdelena na 2 casti
write_vertices_to_csv(convert_to_point_list(knn_e_combined(similarity_mat,0.8,5)), "knn_combined.csv")
# dataset by mela by rozdelena na minimalne 2 velke casti