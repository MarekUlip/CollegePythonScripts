from copy import deepcopy
import random
class Node:
    def __init__(self, parent,val=None):
        self.parent = parent
        self.left_branch = None
        self.right_branch = None
        self.node_val = val

def get_leaves(tree_root:Node,leaves):
    if tree_root.left_branch is not None and tree_root.right_branch is not None:
        print(tree_root.node_val)
        get_leaves(tree_root.left_branch,leaves)
        get_leaves(tree_root.right_branch,leaves)
    else:
        leaves.append(tree_root)
        print("leaf {}".format(tree_root.node_val))


def name_inner_nodes(leaves):
    new_leaves = []
    parsinomy_len = 0
    while len(leaves)>0:
        for leaf in leaves:
            parent = leaf.parent
            if parent is None:
                new_leaves = []
                break
            if parent.node_val is None:
                if parent.left_branch.node_val is None or parent.right_branch.node_val is None:
                    continue
                leaf_vals = parent.left_branch.node_val
                leaf_vals = [item for item in leaf_vals if item in parent.right_branch.node_val]
                if len(leaf_vals) == 0:
                    leaf_vals = deepcopy(parent.left_branch.node_val)
                    leaf_vals.extend(parent.right_branch.node_val)
                    #leaf_vals = parent.left_branch.node_val.extend(parent.right_branch.node_val)
                    parsinomy_len += 1
                parent.node_val = leaf_vals
                new_leaves.append(parent)
        leaves = new_leaves
    print(parsinomy_len)
    return parsinomy_len

def prune_inner_node_names(root:Node,prev_char):
    if root.left_branch is None and root.right_branch is None:
        return
    next_char = []
    if prev_char in root.node_val:
        root.node_val = [prev_char]
        next_char = prev_char
    else:
        i = random.randint(0,len(root.node_val)-1)
        next_char = root.node_val[i]
        root.node_val = [next_char]
    prune_inner_node_names(root.left_branch,next_char)
    prune_inner_node_names(root.right_branch,next_char)
    
                


root = Node(None)
root.right_branch = Node(root,['A'])
branch = Node(root)
root.left_branch = branch
branch2 = Node(branch)
branch.left_branch = branch2
branch2.left_branch = Node(branch2,['C'])
branch2.right_branch = Node(branch2,['T'])

branch2 = Node(branch)
branch.right_branch = branch2
branch2.right_branch = Node(branch2,['A'])
branch = Node(branch2)
branch2.left_branch = branch
branch.left_branch = Node(branch,['G'])
branch.right_branch = Node(branch,['T'])

leaves = []
get_leaves(root,leaves)
name_inner_nodes(leaves)
get_leaves(root,[])
print()
prune_inner_node_names(root,'A')
get_leaves(root,[])

