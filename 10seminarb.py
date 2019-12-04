import numpy as np

# Define labels for the dataset header
headers = [['A'], ['B'], ['C'], ['D'], ['E']]

# Create frame-based difference matrix
"""matrix = [
    [],                         
    [10],                       
    [16, 16],                   
    [26, 26, 26],                
    [20, 26, 26, 26]    
    ]"""
matrix = [
    [],                         
    [20],                       
    [26, 26],                   
    [26, 26, 16],                
    [26, 26, 16, 10]    
    ]

def find_lowest_val_pos(matrix):
    min_val = float('inf')
    pos_x = -1
    pos_y = -1
    
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if(matrix[i][j] < min_val):
                min_val = matrix[i][j]
                pos_x = i
                pos_y = j
                
    return pos_x, pos_y

def join_headers(headers, a, b):
    if b < a:
        a, b = b, a
        
    # Join labels in the first position
    headers[a].extend(headers[b])#"(" + headers[a] + "," + headers[b] + ")"
    key_name = " ".join(item for item in headers[a])
    # Delete label from second position
    del headers[b]
    return key_name

def join_matrix(matrix, a, b):
    if b < a:
        a, b = b, a

    row = []
    for i in range(0, a):
        row.append((matrix[a][i] + matrix[b][i])/2)
    matrix[a] = row
    
    for i in range(a+1, b):
        matrix[i][a] = (matrix[i][a] + matrix[b][i])/2
        
    for i in range(b+1, len(matrix)):
        matrix[i][a] = (matrix[i][a] + matrix[i][b])/2
        del matrix[i][b]

    del matrix[b]

def upgma(matrix, headers):
    print('Matrix of Initial Differences: ', matrix)
    nodes = {}
    edges = []
    iteration = 0
    while len(headers) > 1:        
        print('\nIteration: ', iteration)
        print('#Clusters: ', len(matrix))
                
        # Locate the position in the matrix containing the lowest value
        x, y = find_lowest_val_pos(matrix)
        print('Position with lower value:', x, ',', y, '-> Valor = ',  matrix[x][y])
        
        #pick headers to be joined. get their heights. add height to new node for each of two nodes
        node1 = " ".join(headers[x])
        node2 = " ".join(headers[y])
        h1 = nodes.get(node1,0)
        h2 = nodes.get(node2,0)
        new_height = matrix[x][y]/2
        
        

        key_name = join_headers(headers, x, y)        
        print('Labels updated:', headers)
        edges.append([node1,key_name,new_height-h1])
        edges.append([node2,key_name,new_height-h2])
        nodes[key_name] = new_height
        
        join_matrix(matrix, x, y)
        print('Array values joined:', matrix)
        iteration += 1

    # Final result stored in the first position
    print(headers)
    print(nodes)
    print(edges)
    return headers[0]

print(upgma(matrix, headers))


