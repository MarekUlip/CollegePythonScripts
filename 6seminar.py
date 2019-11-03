from sklearn import preprocessing
import numpy as np
import random
import time

class Node:

    def __init__(self):
        self.true_branch:Node = None
        self.false_branch:Node = None
        self.rule = []
        self.class_num = None
    
    def add_true_branch(self):
        self.true_branch = Node()
        return self.true_branch

    def add_false_branch(self):
        self.false_branch = Node()
        return self.false_branch
    
    def set_rule(self, rule):
        self.rule = rule

    def print_tree(self):
        print(self.rule)
        print(self.class_num)
        print('going to branches')
        if self.false_branch is not None:
            self.false_branch.print_tree()
        if self.true_branch is not None:
            self.true_branch.print_tree()
    
    def go_to_next_rule(self,item):
        if self.class_num is not None:
            return self.class_num
        if self.rule[2]: #is lesser
            if item[self.rule[1]] < self.rule[0]:
                if self.true_branch is None:
                    return self.rule[4]
                else:
                    return self.true_branch.go_to_next_rule(item)
            else:
                return self.false_branch.go_to_next_rule(item)
        else: #is greater
            if item[self.rule[1]] >= self.rule[0]:
                if self.true_branch is None:
                    return self.rule[4]
                else:
                    return self.true_branch.go_to_next_rule(item)
            else:
                return self.false_branch.go_to_next_rule(item)


def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    return x_scaled

def create_decision_treee(datas, feature_indexes:list, target_index,rules:list):
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
        rules.append([best_split,best_split_index, is_lesser, False,get_highest_class(new_branch_data,[],target_index)])
        create_decision_tree(new_branch_data,feature_indexes,target_index,rules)
        create_decision_tree(true_b_data,feature_indexes,target_index,rules)
    #print(rules)

def create_decision_tree(datas, feature_indexes:list, target_index,tree:Node, class_count):
    #num_of_features = len(feature_indexes)
    if len(datas) == 1:
        tree.class_num = datas[0][target_index]
        #print('returning')
        return
    best_gini = 1
    best_split = 0
    best_branch_size = 0
    best_split_index = 0
    other_gini = 1
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
            gini_l = gini_index(create_temp_groups(lesser,target_index, class_count),branch_size_l)
            gini_g = gini_index(create_temp_groups(greater,target_index, class_count),branch_size_g)
            if gini_l <= best_gini and best_branch_size < branch_size_l:
                best_gini = gini_l
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_l
                other_gini = gini_g
                is_lesser=True
            if gini_g <= best_gini and best_branch_size < branch_size_g:
                best_gini = gini_g
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_g
                other_gini = gini_l
                is_lesser = False
    lesser_branch,greater_branch = split_datas(best_split_index,best_split,datas)
    if best_gini == 0:
        if is_lesser:
            tree.rule = [best_split,best_split_index, is_lesser,True, lesser_branch[0][target_index],"contains {}".format(len(lesser_branch))]
            if len(greater_branch) > 0:
                if other_gini == 0:
                    new_node = Node()
                    new_node.class_num = greater_branch[0][target_index]
                    tree.add_false_branch()
                    tree.false_branch = new_node
                    return
                else:
                    create_decision_tree(greater_branch,feature_indexes,target_index,tree.add_false_branch(),class_count)
            else:
                return
        else:
            tree.rule = [best_split,best_split_index, is_lesser,True, greater_branch[0][target_index],"contains {}".format(len(lesser_branch))]
            if len(lesser_branch) > 0:
                if other_gini == 0:
                    new_node = Node()
                    new_node.class_num = lesser_branch[0][target_index]
                    tree.add_false_branch()
                    tree.false_branch = new_node
                    return
                else:
                    create_decision_tree(lesser_branch,feature_indexes,target_index,tree.add_false_branch(),class_count)
            else:
                return
    else:
        print('not the best gini')
        #print(best_gini)
        #print(other_gini)
        whole_level = greater_branch[:].extend(lesser_branch)
        tree.rule = [best_split,best_split_index, is_lesser, False,get_highest_class(greater_branch,lesser_branch,target_index)]
        if is_lesser:
            create_decision_tree(greater_branch,feature_indexes,target_index,tree.add_false_branch(),class_count)
            create_decision_tree(lesser_branch,feature_indexes,target_index,tree.add_true_branch(),class_count)
        else:
            create_decision_tree(greater_branch,feature_indexes,target_index,tree.add_true_branch(),class_count)
            create_decision_tree(lesser_branch,feature_indexes,target_index,tree.add_false_branch(),class_count)
    #print(rules)

#def evaluate()

def get_highest_class(data1,data2, target_index):
    counts = {}
    for row in data1:
        index = int(row[target_index])
        counts[index] = counts.get(index,0) +1
    for row in data2:
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

def create_temp_groups(datas, target_index,class_count, mapping=None):
    groups = [[] for i in range(class_count)]
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

def test_decision_tree(tree:Node,test_set, num_of_classes, target_index):
    conf_matrix = np.zeros((num_of_classes,num_of_classes))
    print('testiong')
    for row in test_set:
        res = int(tree.go_to_next_rule(row))
        print(res)
        conf_matrix[int(row[target_index]),res]+= 1
    return conf_matrix

#Rozdelit si sadu a ukazat jaka je presnost pomoci confusion matrix
arr = np.loadtxt('sep.csv',delimiter=';')
target_index = 2
split_index = 90
class_count = 2
classes = arr[:,target_index].astype(int)
classes = np.reshape(classes,(1,len(classes)))
arr = normalize(arr[:,:target_index])
arr = np.append(arr,np.transpose(classes),axis=1)
arr = arr.tolist()
random.shuffle(arr)
features = [i for i in range(len(arr[0])-1)]
tree = Node()
create_decision_tree(arr[:split_index],features,len(arr[0])-1,tree,class_count)
print(test_decision_tree(tree,arr[split_index:],class_count,len(arr[0])-1))
#tree.print_tree()
#print(len(rules))
#print(create_mapping(arr,4))
#print(normalize(arr[:,:4]))