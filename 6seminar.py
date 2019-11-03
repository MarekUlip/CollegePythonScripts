from sklearn import preprocessing
import numpy as np

class Node:

    def __init__(self):
        self.true_branch:Node = None
        self.false_branch:Node = None
        self.rule = []
    
    def add_true_branch(self):
        self.true_branch = Node()
        return self.true_branch

    def add_false_branch(self):
        self.false_branch = Node()
        return self.false_branch
    
    def set_rule(self, rule):
        self.rule = rule
    
    def go_to_next_rule(self,item):
        if self.rule[2]: #is lesser
            if item[self.rule[1]] < self.rule[0]:
                if self.true_branch is None:
                    return self.rule[4]
                else:
                    return self.false_branch.go_to_next_rule(item)
        else: #is greater
            if item[self.rule[1]] > self.rule[0]:
                if self.true_branch is None:
                    return self.rule[4]
                else:
                    return self.false_branch.go_to_next_rule(item)


def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    return x_scaled

def create_decision_tree(datas, feature_indexes:list, target_index,rules:list):
    #num_of_features = len(feature_indexes)
    if len(datas) == 1:
        return
    best_gini = 1
    best_split = 0
    best_branch_size = 0
    best_split_index = 0
    is_lesser = None
    print(len(datas))
    for i in range(len(feature_indexes)):
        for j in np.arange(0.0, 1.0, 0.1):
            lesser, greater = split_datas(i,j,datas)
            if len(lesser) == 0 or len(greater) == 0:
                continue
            #print(lesser)
            branch_size_l = len(lesser)
            branch_size_g = len(greater)
            gini_l = gini_index(create_temp_groups(lesser,target_index),branch_size_l)
            gini_g = gini_index(create_temp_groups(greater,target_index),branch_size_g)
            if gini_l <= best_gini and best_branch_size < branch_size_l:
                best_gini = gini_l
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_l
                is_lesser=True
            if gini_g <= best_gini and best_branch_size < branch_size_g:
                best_gini = gini_g
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_g
                is_lesser = False
    true_b_data,new_branch_data = split_datas(best_split_index,best_split,datas)
    if best_gini == 0:
        rules.append([best_split,best_split_index, is_lesser,True, true_b_data[0][target_index],"contains {}".format(len(true_b_data))])
        #print("contains {}".format(len(true_b_data)))
        if len(new_branch_data) > 0:
            create_decision_tree(new_branch_data,feature_indexes,target_index,rules)
        else:
            return
    elif len(new_branch_data) == 1:
        print('so this happened and other branch has {}'.format(len(true_b_data)))
        rules.append([best_split,best_split_index, is_lesser,True,new_branch_data[0][target_index]])
        return 
    else:
        print('not the best gini')
        rules.append([best_split,best_split_index, is_lesser, False,get_highest_class(new_branch_data,target_index)])
        create_decision_tree(new_branch_data,feature_indexes,target_index,rules)
        create_decision_tree(true_b_data,feature_indexes,target_index,rules)
    #print(rules)

#def evaluate()

def get_highest_class(datas, target_index):
    counts = {}
    for row in datas:
        index = int(row[target_index])
        counts[index] = counts.get(index,0) +1
    highest = 0
    h_class = -1
    for key, value in counts.items():
        if value> highest:
            highest = value
            h_class = key
    return h_class  
         
            

def gini_index(groups, dataset_size):
    if dataset_size == 0:
        return 0
    sum_all = 0
    for group in groups:
        proportion = len(group)/dataset_size
        sum_all+=proportion**2
    return 1-sum_all

def create_temp_groups(datas, target_index, mapping=None):
    groups = [[] for i in range(target_index-1)]
    for row in datas:
        groups[int(row[target_index])].append(row)
    return groups

def create_mapping(datas, target_index):
    data_set = np.unique((datas[:,target_index]))
    mapping = {}
    for index,item in enumerate(data_set):
        mapping[item] = index
    return mapping


def split_datas(index, value, data):
    lesser = []
    greater = []
    for row in data:
        if row[index]<value:
            lesser.append(row)
        else:
            greater.append(row)
    return lesser, greater

#Rozdelit si sadu a ukazat jaka je presnost pomoci confusion matrix
arr = np.loadtxt('iris.csv',delimiter=';')
classes = arr[:,4].astype(int)
classes = np.reshape(classes,(1,len(classes)))
arr = normalize(arr[:,:4])
arr = np.append(arr,np.transpose(classes),axis=1)
features = [i for i in range(len(arr[0])-1)]
rules = []
create_decision_tree(arr,features,len(arr[0])-1,rules)
print(rules)
print(len(rules))
#print(create_mapping(arr,4))
#print(normalize(arr[:,:4]))