import csv
import math
import statistics
import numpy

dataset = []
with open('weather.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        dataset.append(row)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def get_all_attributes_from_index(index, datas):
        """
            Returns list of all attributes from specified index. If datas are not provided it will
            use dataset asociated with this class.
        """
        atttributes = []
        for item in datas:
            if isfloat(item[index]):
                atttributes.append(float(item[index]))
        return atttributes

def get_all_attributes_from_indexes(indexes, datas):
    attributes = []
    for data in datas:
        attributes.append([float(x) for index, x in enumerate(data) if index in indexes])
    return attributes

def euklid_distance_points(a, mean):
    if type(a) is not list:
        print("converting")
        a = [a]
    return math.sqrt(sum([(x-y)**2 for x,y in zip(a,mean)]))
    
def euclidean_norm(vector):
    return math.sqrt(sum(x**2 for x in vector))

def mean(datas): 
    return sum(datas)/len(datas)

def variance(attributes, are_more_attributes=True):
    m = [mean(attributes)]
    return sum([(euklid_distance_points(x,m))**2 for x in attributes])/(len(attributes)-1)


def stddev(data):
    """
        Calculates the sample standard deviation
    """
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = variance(data)*(len(data)-1)
    pvar = ss/(n-1)
    return pvar**0.5

#print(variance(get_all_attributes_from_indexes([2,1],dataset)))
Covariance = numpy.cov(get_all_attributes_from_index(1,dataset), get_all_attributes_from_index(2,dataset), ddof=1)
print(Covariance)
"""print(mean(get_all_attributes_from_index(2,dataset)))
print(stddev(get_all_attributes_from_index(2,dataset),1))
print(statistics.stdev(get_all_attributes_from_index(2,dataset)))"""