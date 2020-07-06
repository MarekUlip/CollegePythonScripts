"""
Script that implements CFTree.
"""
from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import Birch
from sklearn.datasets.samples_generator import make_blobs


class CFTree:
    def __init__(self, B, T, L, max_num_of_entries=500):
        self.branching_factor = None
        self.threshold = None
        self.max_entries = None
        self.root_node: CFNode = None
        self.items: list = None
        self.index = 0
        self.max_num_of_entries = max_num_of_entries
        self.num_of_entries = 0
        self.dummy = None
        self.rebuild_tree(B, T, L)

    def rebuild_tree(self, B, T, L, max_num_of_entries=500, cf_entries=None):
        self.branching_factor = B
        self.threshold = T
        self.max_entries = L
        self.max_num_of_entries = max_num_of_entries
        self.num_of_entries = 0
        self.dummy = CFNode()
        if cf_entries is not None:
            self.root_node = self.create_root_node()
            for entry in cf_entries:
                self.root_node.insert_entry(entry)
        if self.items is not None:
            self.build_tree(self.items)

    def build_tree(self, items):
        self.items = items
        self.root_node = self.create_root_node()
        item = self.items.pop(0)
        while len(items) >= 0:
            self.root_node.insert_entry(CFEntry(item, self.index))
            if len(items) == 0:
                break
            item = self.items.pop(0)
            self.index += 1
            # print(self.index)

    def get_last_leaf(self):
        node = self.dummy
        while node.next is not None:
            node = node.next
        return node

    def remove_leaf(self, leaf):
        node = None
        while node.next is not leaf:
            node = node.next
        node.prev.next = node.next
        node.next.prev = node.prev

    def replace_leaf_with_leaves(self, leaf, leaves):
        node = self.dummy
        print('replacing')
        while node is not leaf:
            node = node.next
        leaves[0].next = leaves[1]
        leaves[0].prev = node.prev
        leaves[1].next = node.next
        leaves[1].prev = leaves[0]
        node.prev.next = leaves[0]
        if node.next is not None:
            node.next.prev = leaves[1]

    def split_root(self, new_nodes):
        new_root = CFNode(self.root_node.cf_entry, [], tree=self, is_leaf=False)
        for node in new_nodes:
            node.parent = new_root
        new_root.childs.extend(new_nodes)
        self.root_node = new_root

    def create_root_node(self):
        node = CFNode(parent=None, tree=self, is_leaf=False)
        node.cf_entry = CFEntry()
        node2 = CFNode(CFEntry(), parent=node, tree=self, is_leaf=True)
        self.dummy.next = node2
        node2.prev = self.dummy
        node.childs = [node2]
        return node

    def get_cf_entries_from_leaves(self):
        entries = []
        node = self.dummy.next
        while node is not None:
            entries.extend(node.childs)
            node = node.next
        return entries

    def get_clusters(self):
        clusters = []
        node = self.dummy.next
        while node is not None:
            clusters.append(node.childs)
            node = node.next
        return clusters

    def get_point_labels(self):
        clusters = self.get_clusters()
        labels = []
        index = 0
        for cluster in clusters:
            for cf in cluster:
                for item in cf.indexes:
                    labels.append([item, index])
                index += 1
        return np.array(sorted(labels, key=lambda x: x[0]))[:, 1]

    def add_to_leaf_chain(self, node):
        last = self.get_last_leaf()
        node.prev = last
        last.next = node


class CFEntry:
    def __init__(self, point=None, point_index=None, cf_entry=None):
        if point is not None:
            self.N = 1
            self.SS = sum([x ** 2 for x in point])
            self.LS = np.array(point)  # sum(point)
            self.indexes = [point_index]
        elif cf_entry is not None:
            self.N = cf_entry.N
            self.SS = cf_entry.SS
            self.LS = cf_entry.LS
            self.indexes = cf_entry.indexes
        else:
            self.N = 0
            self.SS = 0
            self.LS = 0
            self.indexes = []

    def insert_entry(self, cf_entry):
        self.N += cf_entry.N
        self.SS += cf_entry.SS
        self.LS += cf_entry.LS
        self.indexes.extend(cf_entry.indexes)

    def get_test_radius(self, cf_entry):
        self.N += cf_entry.N
        self.SS += cf_entry.SS
        self.LS += cf_entry.LS
        radius = self.get_radius()
        self.N -= cf_entry.N
        self.SS -= cf_entry.SS
        self.LS -= cf_entry.LS
        return radius

    def get_centroid(self):
        return np.divide(self.LS, self.N)

    def get_radius(self):
        centroid = self.get_centroid()
        r_p_1 = 2.0 * np.dot(self.LS, centroid)
        r_p_2 = self.N * np.dot(centroid, centroid)
        # return np.sqrt(self.SS/self.N - (self.LS/self.N)**2)
        return ((1.0 / self.N) * (self.SS - r_p_1 + r_p_2)) ** 0.5

    def get_diameter(self):
        diameter_part = self.SS * self.N - 2.0 * np.dot(self.LS, self.LS) + self.SS * self.N
        if diameter_part < 0.000000001:
            return 0
        else:
            return (diameter_part / (self.N * (self.N - 1))) ** 0.5

    def count_distance(self, cfentry):
        distance = 0.0
        c1 = cfentry.get_centroid()
        c2 = self.get_centroid()
        for i in range(0, len(c1)):
            distance += (c1[i] - c2[i]) ** 2.0
        return distance
        # return sum([item - item2 for item, item2 in zip(cfentry.get_centroid(),self.get_centroid())])**2
        # return (cfentry.get_centroid()-self.get_centroid())**2.0#np.sqrt((self.N*cfentry.SS+cfentry.N+self.SS - 2* self.LS*cfentry.LS)/(self.N*cfentry.N))


class CFPair:
    def __init__(self, cfentry1, cfentry2, index1, index2):
        self.pair = [cfentry1, cfentry2]
        self.indexes = [index1, index2]


class CFNode:
    def __init__(self, cf_entry: CFEntry = None, parent=None, tree: CFTree = None, is_leaf=True):
        self.cf_entry: CFEntry = cf_entry
        self.childs: list = []
        if self.cf_entry is not None and self.cf_entry.N != 0:
            self.childs.append(self.cf_entry)
        self.parent: CFNode = parent
        self.is_leaf = is_leaf
        if tree is not None:
            self.prev = tree.dummy
        else:
            self.prev = None
        self.next = None
        self.tree = tree

    def split(self):
        print('splitting')
        if self.is_leaf:
            best_pair = []
            best_max_dist = -1
            for index, child1 in enumerate(self.childs):
                for index2, child2 in enumerate(self.childs):
                    if index == index2:
                        continue
                    dist = child1.count_distance(child2)
                    if dist > best_max_dist:
                        best_max_dist = dist
                        best_pair = CFPair(child1, child2, index, index2)
            """new_parents = [CFNode(cf_entry=best_pair.pair[0], is_leaf=self.is_leaf), CFNode(cf_entry=best_pair.pair[1],is_leaf=self.is_leaf)]
            for index, child in enumerate(self.childs):
                if index in best_pair.indexes:
                    continue
                distances = [item.count_distance(child) for item in best_pair.pair]
                new_parents[distances.index(min(distances))].insert_entry(child)
            if self.parent is None:
                self.tree.split_root(new_parents)
            else:
                self.parent.split_node(self,new_parents)"""
        else:
            best_pair = []
            best_max_dist = -1
            for index, child1 in enumerate(self.childs):
                for index2, child2 in enumerate(self.childs):
                    if index == index2:
                        continue
                    dist = child1.cf_entry.count_distance(child2.cf_entry)
                    if dist > best_max_dist:
                        best_max_dist = dist
                        best_pair = CFPair(child1.cf_entry, child2.cf_entry, index, index2)
        new_parents = [CFNode(cf_entry=best_pair.pair[0], parent=self.parent, is_leaf=self.is_leaf, tree=self.tree),
                       CFNode(cf_entry=best_pair.pair[1], parent=self.parent, is_leaf=self.is_leaf, tree=self.tree)]
        for index, child in enumerate(self.childs):
            if index in best_pair.indexes:
                continue
            distances = [item.count_distance(child) for item in best_pair.pair]
            self.tree.num_of_entries -= len(self.childs)
            # start = time()
            new_parents[distances.index(min(distances))].cf_entry.insert_entry(child)
            new_parents[distances.index(min(distances))].childs.append(child)
            # print('inserting took {}'.format(time()-start))
        if self.parent is None:
            self.tree.split_root(new_parents)
        else:
            self.parent.split_node(self, new_parents)
            # self.is_leaf:
            #   self.tree.replace_leaf_with_leaves(self,new_parents)

    def split_node(self, node_to_split, new_nodes):
        self.childs.remove(node_to_split)
        self.childs.extend(new_nodes)
        if node_to_split.is_leaf:
            self.tree.replace_leaf_with_leaves(node_to_split, new_nodes)
        self.check_size_correctnes()

    def insert_entry(self, cfentry):
        if self.cf_entry is None:
            self.cf_entry = CFEntry()
        self.cf_entry.insert_entry(cfentry)
        if len(self.childs) != 0:
            cf_index = self.find_cf_entry(cfentry)
        else:
            cf_index = -1
        found_entry = None
        if cf_index != -1:
            found_entry = self.childs[cf_index]
            if not self.is_leaf:
                found_entry = found_entry.cf_entry
        else:
            print('compensating')

        if self.is_leaf:
            if found_entry is not None and self.is_insertable(found_entry, cfentry):
                self.childs[cf_index].insert_entry(cfentry)
            else:
                self.childs.append(cfentry)
                self.tree.num_of_entries += 1
                # self.check_size_correctnes()
        else:
            self.childs[cf_index].insert_entry(cfentry)

        """else:
            if self.is_leaf:
                self.childs.append(cfentry)
                self.tree.num_of_entries += 1
                self.check_size_correctnes()"""
        """else:
            node = CFNode(parent=self.parent, tree=self.tree)
            node.insert_entry(cfentry)
            node.prev = self.tree.get_last_leaf()
            self.childs.append(node)"""

    def check_size_correctnes(self):
        pass
        if self.is_leaf:
            if len(self.childs) > self.tree.max_entries:
                pass
                self.split()
        else:
            if len(self.childs) > self.tree.branching_factor:
                pass
                self.split()

        if self.tree.num_of_entries > self.tree.max_num_of_entries:
            pass
            self.tree.rebuild_tree(self.tree.branching_factor, self.tree.threshold * 1.5, self.tree.max_entries, 500,
                                   self.tree.get_cf_entries_from_leaves())

    def is_insertable(self, cfentry_to_test: CFEntry, cfentry_to_insert: CFEntry):
        # start = time()
        # cfentry_to_test = deepcopy(cfentry_to_test)
        # cfentry_to_insert = deepcopy(cfentry_to_insert)
        # cfentry_to_test.insert_entry(cfentry_to_insert)
        # print('insert test took {}'.format(time() - start))
        if cfentry_to_test.get_test_radius(cfentry_to_insert) > self.tree.threshold:
            return False
        return True

    def find_cf_entry(self, cfentry):
        min_dist = 999999999
        min_dist_index = -1
        for index, child in enumerate(self.childs):
            if self.is_leaf:
                if child.N == 0:
                    continue
                dist = child.count_distance(cfentry)
            else:
                if child.cf_entry.N == 0:
                    continue
                dist = child.cf_entry.count_distance(cfentry)
            if dist < min_dist:
                min_dist_index = index
                min_dist = dist
        return min_dist_index


X, clusters = make_blobs(n_samples=10000, centers=250, cluster_std=0.70, random_state=0)
birch = CFTree(50, 1.5, 50)
start = time()
birch.build_tree(X.tolist())
print(time() - start)
labels = birch.get_point_labels()
print(len(set(labels)))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.show()
print('Reference')
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5)
start = time()
brc.fit(X)
print(time() - start)
labels = brc.predict(X)
print(len(set(labels)))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.show()
