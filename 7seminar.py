from operator import itemgetter
import math
import csv

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

def get_diffenreces(matrix, index_to_work_with):
    mean,variance = count_mean_and_variance(matrix,index_to_work_with)
    print("{} {}".format(mean,variance))
    results = []
    for row in matrix:
        results.append([row[index_to_work_with],normal_density_function(row[index_to_work_with],mean,variance)])
    results = sorted(results,key=itemgetter(0))

    normal_differences = []
    for item in results:
        #normal_differences.append([item[0],math.fabs(item[0]-mean)])
        normal_differences.append(math.fabs(item[0]-mean))
    return [normal_differences]

def count_normal_distribution_for_difference(matrix, index_to_work_with,write_to_csv=False):
    differences = get_diffenreces(matrix,index_to_work_with)
    mean,variance = count_mean_and_variance(matrix,index_to_work_with)
    print("{} {}".format(mean,variance))
    results = []
    for row in matrix:
        results.append([row[index_to_work_with],normal_density_function(row[index_to_work_with],mean,variance)])
    results = sorted(results,key=itemgetter(0))


    with open("iris-normal-difference.csv", mode='w+', newline='') as stats_file:
            csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in results:
                csv_writer.writerow(item)
    return results

def normal_density_function(x, mean, variance):
    return((1/math.sqrt(2*math.pi*variance)) * math.exp( (-((x-mean)**2))/(2*variance) ) )

def count_mean_and_variance(matrix, index_to_work_with):
    attributes= []
    mean = 0
    for row in matrix:
        attributes.append(row[index_to_work_with])
    mean = sum(attributes)/len(attributes)
    variance = sum([(x-mean)**2 for x in attributes])/len(attributes)
    return (mean, variance)

count_normal_distribution_for_difference(dataset,2)