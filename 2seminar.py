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

def euklid_distance(vec1, vec2):
    return math.sqrt( sum( [(vec1[i] - vec2[i])**2 for i in range(len(vec1))] ))



def create_matrix(data):
    for x in data:
        for y in x:
            if not isfloat(y):
                x.remove(y)
            else:
                
                y = float(y)
    return data



"""vecs = [x for x in matrix]
    #vecs = [x for x in vecs if x is ]
    print(vecs[0])
    n = len(matrix)*(len(vecs)-1)
    return sum([vecs[i] for i in range(len(vecs)-1)]) / n #sum([sum([x for x in vec]) for vec in matrix)/len(matrix)"""

def mean(matrix): 
    n = len(matrix)
    mean = []
    for i in range(len(matrix[0])):
        s = []
        for j in range(n):
            s.append(matrix[j][i])
        mean.append(sum(s)/n)
    return mean

    #return sum(matrix) / len(matrix)
def euclidean_norm(vector):
    return math.sqrt(sum(x**2 for x in vector))

def euklid_distance_points(a, mean):
    """n = len(a)
    s = 0
    t = sum([(x-y)**2 for x,y in zip(a,mean)])
    print(t)"""
    """for i in range(n):
        s += (a[i] - mean[i])**2"""
    #print(s)
    return math.sqrt(sum([(x-y)**2 for x,y in zip(a,mean)]))
    #return math.sqrt( (a-mean)**2)

def variance(matrix):
    # eulidovska vzdalenost na druhou
    print(matrix)
    m = mean(matrix)
    return sum([(euklid_distance_points(x,m))**2 for x in matrix])/len(matrix)

def cosine_similarity(vec_a, vec_b):
    e_a = euclidean_norm(vec_a)
    e_b = euclidean_norm(vec_b)
    vec_a = [x/e_a for x in vec_a]
    vec_b = [x/e_b for x in vec_b]
    return sum(x*y for x,y in zip(vec_a,vec_b))

def cos_and_euc_all(matrix):
    for index, x in enumerate(matrix):
        for i, y in enumerate(matrix):
            if x==y:
                continue
            print("{} and {} have {} euclidean distance and {} cosine similarity".format(index,i,euklid_distance(x,y),cosine_similarity(x,y)))

def show_mean_and_variance_by_attribute():
    for key in dataset_dict.keys():
        print("{}: \nMean: {}\nTotal variance: {}\n".format(key,mean(dataset_dict[key]),variance(dataset_dict[key])))
    

#create_matrix(dataset)
clear_matrix = dataset#create_matrix(dataset)
#print(clear_matrix)
print("Whole dataset: \nMean: {}\nTotal variance: {}\n".format(mean(clear_matrix),variance(clear_matrix)))
#print(mean(clear_matrix))
#print(variance(clear_matrix))
#print(cosine_similarity(clear_matrix[0],clear_matrix[2]))
show_mean_and_variance_by_attribute()
#print(dataset_dict)
#cos_and_euc_all(clear_matrix)
