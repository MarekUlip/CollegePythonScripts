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
        if row[petal_name] not in dataset_dict:
            dataset_dict[row[petal_name]] = []
        r = []
        for item in row:
            if isfloat(item):
                r.append(float(item))
        dataset_dict[row[petal_name]].append(r)
        dataset.append(r)

def create_matrix(data):
    for x in data:
        for y in x:
            if not isfloat(y):
                x.remove(y)
            else:
                
                y = float(y)
    return data

def mean(matrix): 
    n = len(matrix)
    mean = []
    for i in range(len(matrix[0])):
        s = []
        for j in range(n):
            s.append(matrix[j][i])
        mean.append(sum(s)/n)
    return mean

def count_absolute_relative_cumulative(data):
    absolute = {}
    for item in data:
        if item not in absolute:
            absolute[item] = 1
        else:
            absolute[item] += 1
    relative = absolute.copy()
    for key, value in relative.items():
        relative[key] = value/len(data)
    cummulative = {}
    sum = 0
    for key in sorted(absolute):
        cummulative[key] = absolute[key] + sum
        sum+=absolute[key]
    for key in sorted(absolute):
        print("{} - absolute: {}, relative: {}, cummulative: {}".format(key,absolute[key],relative[key],cummulative[key]))
    """print(sorted(absolute))
    print("********")
    print(relative)
    print("********")
    print(cummulative)"""

def count_mean_and_variance(matrix, index_to_work_with):
    attributes= []
    mean = 0
    for row in matrix:
        attributes.append(row[index_to_work_with])
    mean = sum(attributes)/len(attributes)
    variance = sum([(x-mean)**2 for x in attributes])/len(attributes)
    return (mean, variance)

def count_normal_distribution(matrix, index_to_work_with, list_to_save):
    mean,variance = count_mean_and_variance(matrix,index_to_work_with)
    print("{} {}".format(mean,variance))
    results = []
    for row in matrix:
        results.append([row[index_to_work_with],normal_density_function(row[index_to_work_with],mean,variance)])
    results = sorted(results,key=itemgetter(0))
    results.insert(0,"Normal for {}".format(index_to_work_with))
    list_to_save.append(results)
    """with open("iris-index-{}-normal.csv".format(index_to_work_with), mode='w+', newline='') as stats_file:
            csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in results:
                csv_writer.writerow(item)"""

def count_cumulative_distribution(matrix, index_to_work_with, list_to_save):
    attributes = []
    for row in matrix:
        attributes.append(row[index_to_work_with])
    a = min(attributes)
    b = max(attributes)
    rng = np.linspace(0,10,500)
    res = ["Cumulative for {}".format(index_to_work_with)]
    for i in rng:
        if i < a:
            res.append([i,0])
        elif i < b:
            res.append([i,(float((i-a))/float((b-a)))])
        else:
            res.append([i,1])
    res.append("***********")
    attributes = sorted(attributes)
    for i in attributes:
        if i < a:
            res.append([i,0])
        elif i < b:
            res.append([i,(float((i-a))/float((b-a)))])
        else:
            res.append([i,1])
    """with open("iris-index-{}-cumulative.csv".format(index_to_work_with), mode='w+', newline='') as stats_file:
            csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in res:
                csv_writer.writerow(item)"""
    list_to_save.append(res)


def normal_density_function(x, mean, variance):
    return((1/math.sqrt(2*math.pi*variance)) * math.exp( (-((x-mean)**2))/(2*variance) ) ) 

def do_binominal(treshold, index_to_work_with, num_of_observations, num_of_trials):
    matrix = dataset.copy()
    count_of_one = 0
    mean = 0
    atributes = []
    for row in matrix:
        atributes.append(row[index_to_work_with])
        mean+=row[index_to_work_with]
        if row[index_to_work_with] < treshold:
            row[index_to_work_with] = 0
        else:
            row[index_to_work_with] = 1
            count_of_one += 1
    mean /= len(matrix)
    variance = sum([(x-mean)**2 for x in atributes])/len(atributes)
    print("1 is occured {}. times. Mean is {}. Variance is {}.".format(count_of_one, mean, variance))
    return matrix

def count_mean(data):
    dimension = len(data[0])
    mean = []
    for i in range(dimension):
        mean.append(sum([x[i] for x in data])/len(data))
    print(mean)

def get_all_attributes_from_index(data, index):
    atttributes = []
    for item in data:
        atttributes.append(item[index])
    return atttributes

def k_means_one_dim(k,data, minimum_cluster_change):
    clusters = [[] for x in range(k)]
    centroids = []
    for i in range(k):
        centroids.append(data[random.randint(0,len(data))])
    for item in data:
        if item not in centroids:
            closest_centroid = [0,99999999999]
            for index, centroid in enumerate(centroids):
                #clusters[random.randint(0,k)].append(item)
                distance = math.fabs(item-centroid)
                if distance < closest_centroid[1]:
                    closest_centroid[0] = index
                    closest_centroid[1] = distance
            clusters[closest_centroid[0]].append(item)
            
    prev_centroids = centroids.copy()
    for index, cluster in enumerate(clusters):
        centroids[index] = sum(cluster)/len(cluster)
    centroid_change = 0
    for index, centroid in enumerate(centroids):
        centroid_change += math.fabs(centroid-prev_centroids[index])

    while centroid_change > minimum_cluster_change:
        clusters = [[] for x in range(k)]
        for item in data:
            closest_centroid = [0,99999999999]
            for index, centroid in enumerate(centroids):
                #clusters[random.randint(0,k)].append(item)
                distance = math.fabs(item-centroid)
                if distance < closest_centroid[1]:
                    closest_centroid[0] = index
                    closest_centroid[1] = distance
            clusters[closest_centroid[0]].append(item)
            print("Clusters:")
            [print(sorted(cluster)) for cluster in clusters]
        prev_centroids = centroids.copy()
        for index, cluster in enumerate(clusters):
            centroids[index] = sum(cluster)/len(cluster)
        centroid_change = 0
        for index, centroid in enumerate(centroids):
            centroid_change += math.fabs(centroid-prev_centroids[index])
    print("Clusters:")
    [print(sorted(cluster)) for cluster in clusters]
    print("Centroids:")
    print(centroids)

def do_empirical(data, index):
    absolute = {}
    for item in data:
        if item not in absolute:
            absolute[item] = 1
        else:
            absolute[item] += 1
    cummulative = {}
    sum = 0
    for key in sorted(absolute):
        cummulative[key] = absolute[key] + sum
        sum+=absolute[key]
    empirical = []
    for key in sorted(cummulative):
        empirical.append([key, cummulative[key]/len(data)])
    print(empirical)
    with open("iris-empirical-distributions{}.csv".format(index), mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in empirical:
            csv_writer.writerow(item)
    
        

        
    
    

#[print(row) for row in do_binominal(7,0,2,10)]
#do_binominal(7,0,2,10)
#print(count_mean_and_variance(dataset.copy(),0))
#count_normal_distribution(dataset.copy(),0)
#count_cumulative_distribution(dataset.copy(),0)

#print([x[0] for x in dataset])

#count_mean(dataset)
k_means_one_dim(3,get_all_attributes_from_index(dataset,0),0.01)
#count_absolute_relative_cumulative(get_all_attributes_from_index(dataset,2))
distributions = []
"""for i in range(4):    
    #count_normal_distribution(dataset.copy(),i,distributions)
    count_cumulative_distribution(dataset.copy(),i,distributions)"""

"""for i in range(4):
    do_empirical(get_all_attributes_from_index(dataset,i),i)"""

"""with open("iris-cummulative-distributions.csv", mode='w+', newline='') as stats_file:
    csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for distribution in distributions:
        for item in distribution:
            csv_writer.writerow(item)"""