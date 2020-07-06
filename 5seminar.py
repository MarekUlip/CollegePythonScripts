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

#print(matrix)

#SNAP - databaze siti od Jure Leskovec

def clustering_koeficient(vertice):
    vertice_neigh = []
    for index, item in enumerate(matrix[vertice-1]):
        if item == 1:
            vertice_neigh.append(index+1)
    neighbours = sum([x for x in matrix[vertice-1]])
    vertex = []
    for vert in vertice_neigh:
        for vert2 in vertice_neigh:
            if vert == vert2 or ((vert2, vert) in vertex):
                continue
            vertex.append((vert,vert2))
    num_of_edges = 0
    for v in vertex:
        if matrix[v[0]-1][v[1]-1] == 1:
            num_of_edges += 1
    if neighbours*(neighbours-1) == 0:
        return -1
    clstr_koef = (2*num_of_edges)/(neighbours*(neighbours-1))
    return clstr_koef

clusters = []
distribution = []
for i in range(1,35):
    res = clustering_koeficient(i)
    if res == -1:
        continue
    clusters.append(res)
    distribution.append([i,sum([x for x in matrix[i-1]]), res])
    print("{}. {}".format(i,clustering_koeficient(i)))


print("Average is {}".format(sum([x for x in clusters])/len(clusters)))
avg_res = {}
for vertice in distribution:
    if vertice[1] not in avg_res:
        avg_res[vertice[1]] = [1, vertice[2]]
    else:
        avg_res[vertice[1]] = [avg_res[vertice[1]][0]+1, avg_res[vertice[1]][1] + vertice[2]]

for key, item in sorted(avg_res.items()):
    print("Degree {} has average clustering coeficient {}".format(key,item[1]/item[0]))
#clustering_koeficient(1)