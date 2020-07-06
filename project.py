import csv
from enum import Enum
import math
import numpy as np
import random

class AttributeTypes(Enum):
    TEXT = 0
    LOGICAL = 1
    NUMBER = 2
    UNKNOWN = 3


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

class DatasetAnalyser:
    def __init__(self, dataset_path, missing_char, first_row_as_descriptor, delimeter, output_path = "summary.txt"):
        self.missing_char = missing_char
        self.first_row_as_descriptor = first_row_as_descriptor
        self.attribute_descritpions = None
        self.dataset = self.load_dataset(dataset_path,delimeter)
        if not first_row_as_descriptor:
            self.create_artificial_attributes()
        self.attribute_types = []
        self.dataset_summary = []
        self.output_path = output_path
        self.get_data_types()
        self.fix_missing_values()
        self.do_basic_summary()
        
    def load_dataset(self,path,delimeter):
        dataset = []
        with open(path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=delimeter)
            for row in readCSV:
                r = []
                for item in row:
                    if isfloat(item):
                        r.append(float(item))
                    else:
                        r.append(item)
                dataset.append(r)
        if self.first_row_as_descriptor:
            self.attribute_descritpions = dataset[0]
            del dataset[0]
        return dataset
    
    def print_summary(self):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for line in self.dataset_summary:
                f.write("{}\n".format(line))
    
    def create_artificial_attributes(self):
        self.attribute_descritpions = []
        for i in range(len(self.dataset[0])):
            self.attribute_descritpions.append("Attribute{}".format(i))
    
    def get_data_types(self):
        unknown_indexes = []
        for index, item in enumerate(self.dataset[0]):
            if item == self.missing_char:
                unknown_indexes.append(index)
                self.attribute_types.append(AttributeTypes.UNKNOWN)
            else:
                self.attribute_types.append(self.get_type(item))
            """elif isfloat(item):
                self.attribute_types.append(AttributeTypes.NUMBER)
            elif item.upper() == "TRUE" or item.upper() == "FALSE":
                self.attribute_types.append(AttributeTypes.LOGICAL)
            else:
                self.attribute_types.append(AttributeTypes.TEXT)"""
        if len(unknown_indexes) != 0:
            for index in unknown_indexes:
                for row in self.dataset:
                    if row[index] != self.missing_char:
                        self.attribute_types[index] = self.get_type(row[index])
                        break

    
    def get_type(self, item):
        if isfloat(item):
            return AttributeTypes.NUMBER
        elif item.upper() == "TRUE" or item.upper() == "FALSE":
            return AttributeTypes.LOGICAL
        else:
            return AttributeTypes.TEXT

    def fix_missing_values(self):
        num_of_attributes = len(self.dataset[0])
        for i in range(num_of_attributes):
            missing_indexes = []
            correct_attributes = []
            attrs = self.get_all_attributes_from_index(i)
            for index, attribute in enumerate(attrs):
                if attribute == self.missing_char:
                    missing_indexes.append(index)
                else:
                    correct_attributes.append(attribute)
            if len(missing_indexes) == 0:
                print("No missing attributes found for attribute {}".format(self.attribute_descritpions[i]))
                continue
            else:
                print("There are {} missing attributes for attribute {}".format(len(missing_indexes),self.attribute_descritpions[i]))
            if self.attribute_types[i] == AttributeTypes.NUMBER:
                median = self.median(correct_attributes)
                for index in missing_indexes:
                    self.dataset[index][i] = median
            else:
                occurences = self.count_text_occurences(correct_attributes)
                best = occurences[0][0]
                for index in missing_indexes:
                    self.dataset[index][i] = best                


    def do_basic_summary(self):
        nrow = len(self.dataset)
        ncol = len(self.dataset[0])
        self.dataset_summary.append("Dataset contains {} objects and {} attributes".format(nrow,ncol))
        for index, attribute in enumerate(self.attribute_descritpions):
            attributes = self.get_all_attributes_from_index(index)
            if self.attribute_types[index] == AttributeTypes.NUMBER:
                mn = self.mean(attributes)
                median = self.median(attributes)
                variance = self.variance(attributes)
                standard_deviation = self.standard_deviation(attributes)
                self.dataset_summary.append("Attribute name: {}\t Type: {}\t Mean: {}\t Median: {}\t Max: {}\t Min: {}\t Variance: {}\t Standard Deviation: {}\t".format(attribute, AttributeTypes.NUMBER.name, mn, median, max(attributes), min(attributes),variance,standard_deviation))



    def mean(self, datas): 
        return sum(datas)/len(datas)
    
    def median(self, datas):
        d = sorted(datas)
        length = len(d)
        if length%2==0:
            return (d[length-1]+d[length])/2
        else:
            return d[math.floor(length/2)]
    
    def euklid_distance_points(self, point_a, point_b):
        if type(point_a) is not list:
            point_a = [point_a]
        if type(point_b) is not list:
            point_b = [point_b]
        if len(point_a) != len(point_b):
            print("Lengths of points do not match. Cannot count distance. Returning")
            return 
        return math.sqrt(sum([(x-y)**2 for x,y in zip(point_a,point_b)]))
    
    def euclidean_norm(self, vector):
        return math.sqrt(sum(x**2 for x in vector))

    def total_variance(self, attributes):
        m = []
        for i in range(len(attributes[0])):
            m.append(self.mean(self.get_all_attributes_from_index(i,attributes)))
        return sum([(self.euklid_distance_points(x,m))**2 for x in attributes])/(len(attributes)-1)
    
    def variance(self,attributes):
        m = [self.mean(attributes)]
        return sum([(self.euklid_distance_points(x,m))**2 for x in attributes])/(len(attributes)-1)

    def standard_deviation(self, data):
        """
            Calculates the sample standard deviation
        """
        n = len(data)
        if n < 2:
            raise ValueError('variance requires at least two data points')
        ss = self.variance(data)*(len(data)-1)
        pvar = ss/(n-1)
        return pvar**0.5

    
    def cosine_similarity(self, vec_a, vec_b):
        """
            Returns cosine similarity between two provided vectors
        """
        e_a = self.euclidean_norm(vec_a)
        e_b = self.euclidean_norm(vec_b)
        vec_a = [x/e_a for x in vec_a]
        vec_b = [x/e_b for x in vec_b]
        return sum(x*y for x,y in zip(vec_a,vec_b))
    
    def count_text_occurences(self, texts):
        occurences = {}
        for text in texts:
            if text not in occurences:
                occurences[text] = 1
            else:
                occurences[text] += 1
        return sorted(occurences.items(), key=lambda kv: kv[1])
    
    def get_all_attributes_from_index(self,index, datas = None):
        """
            Returns list of all attributes from specified index. If datas are not provided it will
            use dataset asociated with this class.
        """
        if datas is None:
            datas = self.dataset
        atttributes = []
        for item in datas:
            atttributes.append(item[index])
        return atttributes
    
    def get_all_attributes_from_indexes(self,indexes, datas = None):
        """
            Returns list of all attributes from specified index. If datas are not provided it will
            use dataset asociated with this class.
        """
        if datas is None:
            datas = self.dataset
        atttributes = []
        for item in datas:
            tmp = []
            for index in indexes:
                tmp.append(item[index])
            atttributes.append(tmp)
        return atttributes

    def find_new_centroids(self, clusters, centroids):
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
    
    def count_centroids_difference(self, centroids, prev_centroids):
        centroid_change = 0
        for index, centroid in enumerate(centroids):
            centroid_change += self.euklid_distance_points(centroid,prev_centroids[index])
        return centroid_change
    
    def create_new_clusters(self, data, centroids, k):
        clusters = [[] for x in range(k)]
        cluster_items_indexes = [[] for x in range(k)]
        for index, item in enumerate(data):
            if item not in centroids:
                closest_centroid = [0,99999999999]
                for index2, centroid in enumerate(centroids):
                    #clusters[random.randint(0,k)].append(item)
                    distance = self.euklid_distance_points(item,centroid)
                    if distance < closest_centroid[1]:
                        closest_centroid[0] = index2
                        closest_centroid[1] = distance
                clusters[closest_centroid[0]].append(item)
                cluster_items_indexes[closest_centroid[0]].append(index)
        return clusters, cluster_items_indexes

    def find_number_attributes(self):
        indexes = []
        for index, attr_type in (self.attribute_types):
            if attr_type is AttributeTypes.NUMBER:
                indexes.append(index)
        return indexes


    def k_means_best(self, minimum_cluster_change):
        data = self.get_all_attributes_from_indexes(self.find_number_attributes())
        clusters = []
        for i in range(2,11):
            clusters.append(self.k_means(data,i,minimum_cluster_change))
        clusters.sort(key = lambda pair: pair[1])
        best_clustering=clusters[0][0]
        best_sse = clusters[0][1]
        for i, cluster in enumerate(best_clustering):
            self.dataset_summary.append("Cluster {}:".format(i))
            for index in cluster:
                self.dataset_summary.append(str(data[index]))
        self.dataset_summary.append("SSE: {}".format(best_sse))

    
    def k_means(self, data, k, minimum_cluster_change):
        #TODO clear clusters after each iteration, make centroid creatin n dimensional
        centroids = []
        for i in range(k):
            centroids.append(data[random.randint(0,len(data))])
        clusters, cluster_items_indexes = self.create_new_clusters(data, centroids, k)
                
        prev_centroids = centroids.copy()
        centroids = self.find_new_centroids(clusters,centroids)
        centroid_change = self.count_centroids_difference(centroids,prev_centroids)

        while centroid_change > minimum_cluster_change:
            clusters, cluster_items_indexes = self.create_new_clusters(data, centroids, k)
            prev_centroids = centroids.copy()
            centroids = self.find_new_centroids(clusters,centroids)
            centroid_change = self.count_centroids_difference(centroids,prev_centroids)

        """self.dataset_summary.append("Clusters:")
        self.dataset_summary.append("".join(map(sorted(clusters),str)))
        self.dataset_summary.append("Centroids:")
        self.dataset_summary.append("".join(map(centroids,str)))
        self.dataset_summary.append("SSE:")
        self.dataset_summary.append(str(self.sum_square_error(clusters,centroids)))"""
        return [cluster_items_indexes, self.sum_square_error(clusters,centroids)]

    
    def sum_square_error(self, clusters, centroids):
        sse = 0
        for index, centroid in enumerate(centroids):
            sse += sum(self.euklid_distance_points(point, centroid) for point in clusters[index])
        return sse
                

def analyse_dataset(dataset_path, missing_char, first_row_as_descriptor, delimeter):
    analyser = DatasetAnalyser(dataset_path,missing_char, first_row_as_descriptor, delimeter)
    analyser.k_means_best(0.000001)
    [print(line) for line in analyser.dataset_summary]

analyse_dataset("testDataIris.csv", "?", False, ',')

