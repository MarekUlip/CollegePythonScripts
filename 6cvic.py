import csv

matrix = [[0 for col in range(34)] for row in range(34)]
nodes = []
edges = []
with open('1KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        edges.append(row.copy())
        row[0] = int(row[0])
        row[1] = int(row[1])
        # original_data.append(row)
        col = int(row[1]) - 1
        row = int(row[0]) - 1
        matrix[row][col] = 1
        matrix[col][row] = 1
        nodes.append(row)


def k_core(k, matrix):
    k_classes = {}
    n = len(matrix[0])
    node_degrees = [sum(row) for row in matrix]
    # for i in range(1,k+1):

    k_classes = []
    removed = 999
    while removed > 0:
        removed = 0
        for vertex, node_degree in enumerate(node_degrees):
            if node_degree < k:
                # matrix[vertex] = [0 for i in range(n)]
                for index, col in enumerate(matrix[vertex]):
                    if col == 1:
                        removed += 1
                        matrix[vertex][index] = matrix[index][vertex] = 0
        node_degrees = [sum(row) for row in matrix]
    print(node_degrees)
    # for index, node_degree in enumerate(node_degrees):
    #    if node_degree >= i:
    #        k_classes[i].append(index)
    for index, degree in enumerate(node_degrees):
        if degree > 0:
            k_classes.append(index)

    """for i in range(1,k):
        #print(len(k_classes[i]))
        to_remove = []
        for item in k_classes[i]:            
            if item in k_classes[i+1]:
                to_remove.append(item)
        for item in to_remove:
            k_classes[i].remove(item)
        #print(counter)"""

    print(k_classes)
    return k_classes


# [print(row) for row in matrix]
k_classes = []
for i in range(1, 5):
    k_classes.append(k_core(i, matrix.copy()))

for clas in k_classes:
    print(clas)
    print(len(clas))
