import csv

dimension = 16
actors_mapping = {}
contacts_ids_mapping = {}


def create_actors_mapping():
    a_id = 0
    with open('florentine-actors.csv') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', "")  # [0:len(line)-1]
            print(line)
            actors_mapping[line] = a_id
            a_id += 1

        print([line.replace('\n', "") for line in lines])


def get_vertices():
    a_id = 0
    with open('1contact_list.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[1] not in contacts_ids_mapping.keys():
                contacts_ids_mapping[row[1]] = a_id
                a_id += 1
            if row[2] not in contacts_ids_mapping.keys():
                contacts_ids_mapping[row[2]] = a_id
                a_id += 1


def load_contact_list():
    day = 0
    prev_time = 0
    dimension = len(contacts_ids_mapping)
    with open('1contact_list.csv') as csvfile:
        matrices = {0: [[0 for col in range(dimension)] for row in range(dimension)],
                    1: [[0 for col in range(dimension)] for row in range(dimension)],
                    2: [[0 for col in range(dimension)] for row in range(dimension)]}
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if int(row[0]) - prev_time > 20000:
                print("Changing day {}".format(prev_time))
                day += 1
            prev_time = int(row[0])
            matrices[day][contacts_ids_mapping[row[1]]][contacts_ids_mapping[row[2]]] = \
            matrices[day][contacts_ids_mapping[row[2]]][contacts_ids_mapping[row[1]]] = 1
    return matrices


def load_multilayer_network():
    with open('florentine.csv') as csvfile:
        matrixes = {"marriage": [[0 for col in range(dimension)] for row in range(dimension)],
                    "business": [[0 for col in range(dimension)] for row in range(dimension)]}
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            matrixes[row[2]][actors_mapping[row[0]]][actors_mapping[row[1]]] = matrixes[row[2]][actors_mapping[row[1]]][
                actors_mapping[row[0]]] = 1
    return matrixes


def join_layers(matrices, num_of_nodes=dimension):
    matrix = [[0 for col in range(num_of_nodes)] for row in range(num_of_nodes)]
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            for item in matrices.values():
                if item[i][j] == 1:
                    matrix[i][j] = 1
                    break
    return matrix


def count_degree_centrality(matrices, actors_mapping):
    centralities = {}
    for actor, a_id in actors_mapping.items():
        degree_centrality = sum([sum(matrix[a_id]) for matrix in matrices.values()])
        print("Degree centrality for actor: {} is {}".format(actor, degree_centrality))
        centralities[a_id] = degree_centrality
        # for matrix in matrices.values():
    return centralities


def count_neighbours(matrix, actors_mapping):
    neighbors = {}
    neighbourhoods = {}
    for actor, a_id in actors_mapping.items():
        neighs = []
        for index, col in enumerate(matrix[a_id]):
            if col > 0:
                for key, value in actors_mapping.items():
                    if value == index:
                        neighs.append(key)
        neighbors[actor] = neighs
        neighbourhoods[actor] = len(neighs)
        print("Neighbours of actor {} are: {}\nNeighbourhood centrality is: {}".format(actor, neighs, len(neighs)))
    return neighbors, neighbourhoods


def count_connective_redundancy(degrees, neighbourhoods, actors_mapping):
    for key, value in actors_mapping.items():
        c_r = 0
        if neighbourhoods[key] == 0 and degrees[value] == 0:
            c_r = 1
        elif neighbourhoods[key] == 0:
            c_r = 1
        elif degrees[value] == 0:
            c_r = "Inf"
        else:
            c_r = 1 - (neighbourhoods[key] / degrees[value])
        print("Connective redundancy for actor {} is {}".format(key, c_r))


# def count_exclusive_centr(matrices):
#    for 


create_actors_mapping()
matrices = load_multilayer_network()

degrees = count_degree_centrality(matrices, actors_mapping)
neighbors, neighbourhoods = count_neighbours(join_layers(matrices, len(actors_mapping)), actors_mapping)

count_connective_redundancy(degrees, neighbourhoods, actors_mapping)

"""get_vertices()
print(contacts_ids_mapping)
matrices = load_contact_list()

degrees = count_degree_centrality(matrices, contacts_ids_mapping)
neighbors, neighbourhoods = count_neighbours(join_layers(matrices, len(contacts_ids_mapping)), contacts_ids_mapping)

count_connective_redundancy(degrees, neighbourhoods, contacts_ids_mapping)"""

# Laplaceova matice a kcemu je
# Jak funguje bianconz
# random walk vyhody nevyhody
# jak funguje hiearchicke shlukovani
