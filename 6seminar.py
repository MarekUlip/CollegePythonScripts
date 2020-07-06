"""
Script implementing decision tree.
"""
import random
import time

import numpy as np
from sklearn import preprocessing


class Node:

    def __init__(self, is_numerical):
        self.true_branch: Node = None
        self.false_branch: Node = None
        self.is_numerical = is_numerical
        self.rule = []
        self.class_num = None

    def add_true_branch(self):
        self.true_branch = Node(self.is_numerical)
        return self.true_branch

    def add_false_branch(self):
        self.false_branch = Node(self.is_numerical)
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

    def go_to_next_rule(self, item):
        if self.class_num is not None:
            return self.class_num
        if self.is_numerical:
            if self.rule[2]:  # is lesser
                if item[self.rule[1]] < self.rule[0]:
                    if self.true_branch is None:
                        return self.rule[4]
                    else:
                        return self.true_branch.go_to_next_rule(item)
                else:
                    return self.false_branch.go_to_next_rule(item)
            else:  # is greater
                if item[self.rule[1]] >= self.rule[0]:
                    if self.true_branch is None:
                        return self.rule[4]
                    else:
                        return self.true_branch.go_to_next_rule(item)
                else:
                    return self.false_branch.go_to_next_rule(item)
        else:
            if self.rule[2]:
                if item[self.rule[1]] == self.rule[0]:
                    if self.true_branch is None:
                        return self.rule[4]
                    else:
                        return self.true_branch.go_to_next_rule(item)
                else:
                    return self.false_branch.go_to_next_rule(item)
            else:
                if item[self.rule[1]] != self.rule[0]:
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


def create_decision_tree(datas, feature_indexes: list, target_index, tree: Node, class_count, is_numerical,
                         possible_values=None):
    # num_of_features = len(feature_indexes)
    if len(datas) == 1:
        tree.class_num = datas[0][target_index]
        # print('returning')
        return
    best_gini = 1
    best_split = 0
    best_branch_size = 0
    best_split_index = 0
    other_gini = 1
    is_lesser = None
    # print(len(datas))
    iter_range = []
    if is_numerical:
        iter_range = np.arange(0.0, 1.0, 0.1)
    else:
        iter_range = possible_values
    for i in range(len(feature_indexes)):
        for j in iter_range:
            lesser, greater = split_datas(i, j, datas, is_numerical)
            branch_size_l = len(lesser)
            branch_size_g = len(greater)
            gini_l = 1
            gini_g = 1
            if branch_size_l == 0 or branch_size_g == 0:
                if branch_size_l > 0:
                    gini_l = gini_index(create_temp_groups(lesser, target_index, class_count), branch_size_l)
                else:
                    gini_g = gini_index(create_temp_groups(greater, target_index, class_count), branch_size_g)
                if gini_l != 0 or gini_g != 0:
                    continue
            else:
                # print(lesser)
                gini_l = gini_index(create_temp_groups(lesser, target_index, class_count), branch_size_l)
                gini_g = gini_index(create_temp_groups(greater, target_index, class_count), branch_size_g)
            if gini_l <= best_gini and best_branch_size < branch_size_l:
                best_gini = gini_l
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_l
                other_gini = gini_g
                is_lesser = True
            if gini_g <= best_gini and best_branch_size < branch_size_g:
                best_gini = gini_g
                best_split = j
                best_split_index = i
                best_branch_size = branch_size_g
                other_gini = gini_l
                is_lesser = False
    lesser_branch, greater_branch = split_datas(best_split_index, best_split, datas, is_numerical)
    if best_gini == 0:
        if is_lesser:
            tree.rule = [best_split, best_split_index, is_lesser, True, lesser_branch[0][target_index],
                         "contains {}".format(len(lesser_branch))]
            if len(greater_branch) > 0:
                if other_gini == 0:
                    new_node = tree.add_false_branch()
                    new_node.class_num = greater_branch[0][target_index]
                    return
                else:
                    create_decision_tree(greater_branch, feature_indexes, target_index, tree.add_false_branch(),
                                         class_count, is_numerical, possible_values)
            else:
                new_branch = tree.add_false_branch()
                c_num = random.randint(0, class_count - 1)
                while c_num == lesser_branch[0][target_index]:
                    c_num = random.randint(0, class_count - 1)
                new_branch.class_num = c_num
                return
        else:
            tree.rule = [best_split, best_split_index, is_lesser, True, greater_branch[0][target_index],
                         "contains {}".format(len(lesser_branch))]
            if len(lesser_branch) > 0:
                if other_gini == 0:
                    new_node = tree.add_false_branch()
                    new_node.class_num = lesser_branch[0][target_index]
                    return
                else:
                    create_decision_tree(lesser_branch, feature_indexes, target_index, tree.add_false_branch(),
                                         class_count, is_numerical, possible_values)
            else:
                new_branch = tree.add_false_branch()
                c_num = random.randint(0, class_count - 1)
                while c_num == greater_branch[0][target_index]:
                    c_num = random.randint(0, class_count - 1)
                new_branch.class_num = c_num
                return
    else:
        # print('not the best gini')
        if best_gini == 1 and other_gini == 1:
            h_class = get_highest_class(greater_branch, lesser_branch, target_index)
            tree.rule = [0.0, 0, True, True, h_class]
            c_num = random.randint(0, class_count - 1)
            while c_num == h_class:
                c_num = random.randint(0, class_count - 1)
            tree.add_false_branch().class_num = c_num
            return
        """print('lLen{} gLen{}'.format(len(lesser_branch),len(greater_branch)))
        best_gini = gini_index(create_temp_groups(lesser_branch,target_index,class_count),len(lesser_branch))
        other_gini = gini_index(create_temp_groups(greater_branch,target_index,class_count),len(greater_branch))
        print('gini {} otherGini {}'.format(best_gini,other_gini))"""
        # print(best_gini)
        # print(other_gini)
        tree.rule = [best_split, best_split_index, is_lesser, False,
                     get_highest_class(greater_branch, lesser_branch, target_index)]
        if is_lesser:
            create_decision_tree(greater_branch, feature_indexes, target_index, tree.add_false_branch(), class_count,
                                 is_numerical, possible_values)
            create_decision_tree(lesser_branch, feature_indexes, target_index, tree.add_true_branch(), class_count,
                                 is_numerical, possible_values)
        else:
            create_decision_tree(greater_branch, feature_indexes, target_index, tree.add_true_branch(), class_count,
                                 is_numerical, possible_values)
            create_decision_tree(lesser_branch, feature_indexes, target_index, tree.add_false_branch(), class_count,
                                 is_numerical, possible_values)
    # print(rules)


# def evaluate()

def get_highest_class(data1, data2, target_index):
    counts = {}
    for row in data1:
        index = int(row[target_index])
        counts[index] = counts.get(index, 0) + 1
    for row in data2:
        index = int(row[target_index])
        counts[index] = counts.get(index, 0) + 1
    highest = 0
    h_class = -1
    for key, value in counts.items():
        if value > highest:
            highest = value
            h_class = key
    return h_class


def gini_index(groups, dataset_size):
    if dataset_size == 0:
        return 0
    sum_all = 0
    for group in groups:
        proportion = len(group) / dataset_size
        sum_all += proportion ** 2
    return 1 - sum_all


def create_temp_groups(datas, target_index, class_count, mapping=None):
    groups = [[] for i in range(class_count)]
    for row in datas:
        groups[int(row[target_index])].append(row)
    return groups


def create_mapping(datas, target_index):
    data_set = np.unique((datas[:, target_index]))
    mapping = {}
    for index, item in enumerate(data_set):
        mapping[item] = index
    return mapping


def split_datas(index, value, data, numerical):
    lesser = []
    greater = []
    for row in data:
        if numerical:
            if row[index] < value:
                lesser.append(row)
            else:
                greater.append(row)
        else:
            if row[index] == value:
                lesser.append(row)
            else:
                greater.append(row)

    return lesser, greater


def test_decision_tree(tree: Node, test_set, num_of_classes, target_index):
    conf_matrix = np.zeros((num_of_classes, num_of_classes))
    print('testiong')
    hits = 0
    for row in test_set:
        res = int(tree.go_to_next_rule(row))
        if res == int(row[target_index]):
            hits += 1
        # print(res)
        conf_matrix[int(row[target_index]), res] += 1
    print('Accuracy is {}'.format(hits / len(test_set)))
    return conf_matrix


# Rozdelit si sadu a ukazat jaka je presnost pomoci confusion matrix
# params = ['sep.csv',2,80,2,True,[]]
# params = ['nonsep.csv',2,112,2,True,[]]
# params = ['iris.csv',4,120,3,True,[]]
# params = ['tic-tac-toe.csv',9,766,2,False,['o','x','b']]
params_list = [
    ['sep.csv', 2, 80, 2, True, []],
    ['nonsep.csv', 2, 112, 2, True, []],
    ['iris.csv', 4, 120, 3, True, []],
    ['tic-tac-toe.csv', 9, 766, 2, False, ['o', 'x', 'b']]]

for params in params_list:
    print('testing dataset {}'.format(params[0]))
    arr = np.loadtxt(params[0], delimiter=';', dtype=str)
    target_index = params[1]
    split_index = params[2]
    class_count = params[3]
    classes = arr[:, target_index].astype(int)
    classes = np.reshape(classes, (1, len(classes)))
    if params[4]:
        arr = normalize(arr[:, :target_index])
        arr = np.append(arr, np.transpose(classes), axis=1)
    arr = arr.tolist()
    random.shuffle(arr)
    features = [i for i in range(len(arr[0]) - 1)]
    tree = Node(params[4])
    start = time.time()
    create_decision_tree(arr[:split_index], features, len(arr[0]) - 1, tree, class_count, params[4], params[5])
    print('Creating took {}'.format(time.time() - start))
    print(test_decision_tree(tree, arr[split_index:], class_count, len(arr[0]) - 1))
    print('--------------------')
    # tree.print_tree()
# print(len(rules))
# print(create_mapping(arr,4))
# print(normalize(arr[:,:4]))
