import math
import csv
import os
import random
from operator import itemgetter
import numpy as np
dataset = []
dataset_dict = {}
# vypocitat euklidovskou vzdalenost od prumeru
# vypocitat pro kazdy atribut zvlast

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

with open('testDataIris.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    petal_name = 4
    for row in readCSV:
        """if row[petal_name] not in dataset_dict:
            dataset_dict[row[petal_name]] = []"""
        r = []
        for item in row:
            if isfloat(item):
                r.append(float(item))
            else:
                r.append(item)
        #dataset_dict[row[petal_name]].append(r)
        dataset.append(r)

def create_matrix(data):
    for x in data:
        for y in x:
            if not isfloat(y):
                x.remove(y)
            else:
                
                y = float(y)
    return data

def euklid_distance(vec1, vec2):
    return math.sqrt( sum( [(vec1[i] - vec2[i])**2 for i in range(len(vec1))] ))

def k_nearest_neighbours(k, item_to_clasify, datas):
    distances = []
    for index, item in enumerate(datas):
        distance = euklid_distance(item[:2],item_to_clasify[:2])
        distances.append([distance,index])
    distances = sorted(distances)
    counts = {}
    for item in distances[:k]:
        key = datas[item[1]][2]
        if key not in counts:
            counts[key] = 1
        else:
            counts[key] += 1
    cs = sorted(counts.items(), key=lambda kv: kv[1])
    return cs[len(cs)-1][0]
    """print(item_to_clasify)
    print(cs)
    print(cs[len(cs)-1][0])"""

def get_all_attributes_from_indexes(data, index1, index2, meta):
    atttributes = []
    for item in data:
        atttributes.append([item[index1],item[index2],item[meta]])
    return atttributes

# udelat to rozdeleni nahodne
def divide_into_groups(matrix, count_of_groups):
    datas = matrix.copy()
    groups = [[] for x in range(count_of_groups)]
    size_of_group = int(len(matrix)/count_of_groups)
    print(size_of_group)
    all = 0
    for i in range(count_of_groups):
        for j in range(size_of_group):
            rnd = random.randint(0,len(datas)-1)
            groups[i].append(datas[rnd])
            datas.remove(datas[rnd])
    return groups

def join_from_groups(groups, index_to_avoid):
    set_to_return = []
    for index, item in enumerate(groups):
        if index == index_to_avoid:
            continue
        for row in item:
            set_to_return.append(row)
    return set_to_return

def test_classifier(groups):
    ks = [x for x in range(1,11)]
    for k in ks:
        print("Testing classifing for k {}".format(k))
        for index, group in enumerate(groups):
            successful = 0
            data_from_groups = join_from_groups(groups,index)
            for item in group:
                result = k_nearest_neighbours(k,item,data_from_groups)
                #print("{} was classified as {}".format(item[2],result))
                if result == item[2]:
                    successful +=1
            print("Testing group {} had {}% successs".format(index,(successful/len(group))*100))
        print()


dataset_reduced = get_all_attributes_from_indexes(dataset,0,2,4)
groups = divide_into_groups(dataset_reduced,10)
#[print(row) for row in divide_into_groups(dataset_reduced,10)]
#data_from_groups = join_from_groups(groups,0)
#k_nearest_neighbours(10,groups[0][0],data_from_groups)
test_classifier(groups)
#print(dataset_reduced[0][:2])



