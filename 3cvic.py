import csv
import random

matrix = [[0 for col in range(34)] for row in range(34)]
nodes = []
with open('1KarateClub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        row[0] = int(row[0])
        row[1] = int(row[1])
        #original_data.append(row)
        col = int(row[1])-1
        row = int(row[0])-1
        matrix[row][col] = 1
        matrix[col][row] = 1
        nodes.append(row)

def keringhan_lin_edges():
    print(len(nodes))
    nds = nodes.copy()
    group1 = []
    group2 = []
    for i in range(int(len(nodes)/2)):
        rnd = random.randint(0,len(nds)-1)
        print(len(nds))
        group1.append(nds[rnd])
        nds.pop(rnd)
        rnd = random.randint(0,len(nds)-1)
        group2.append(nds[rnd])
        nds.pop(rnd)
    if len(nds) > 0:
        for item in nds:
            group1.append(item)
    print("1.: {} 2.: {}".format(len(group1), len(group2)))
    print(group1)
    print(group2)

def keringhan_lin():
    group1 = []
    group2 = []
    base = [i for i in range(34)]
    random.shuffle(base)
    border = len(matrix)//2
    group1 = base[:border]
    group2 = base[border:]
    init_cut_size = calculate_cut_size(0,0,group1, group2, False)
    new_cut_size = 0
    for i in range(10): #while init_cut_size > new_cut_size:
        init_cut_size = new_cut_size#calculate_cut_size(0,0,group1, group2, False)
        restricted = 0#[]
        best_cuts = []
        group_1_for_pick = group1.copy()
        group_2_for_pick = group2.copy()
        while restricted < len(matrix)-1:
            cut_sizes = []
            groups_bck_up = [group_1_for_pick.copy(), group_2_for_pick.copy()]
            while (group_1_for_pick) and (group_2_for_pick):
                i = group_1_for_pick.pop(random.randint(0,len(group_1_for_pick)-1))
                j = group_2_for_pick.pop(random.randint(0,len(group_2_for_pick)-1)) 
                cut_sizes.append([i,j, calculate_cut_size(i,j, group1.copy(), group2.copy())])
            #for i in group1: # znahodnit
                #for j in group2:
                    #if i in restricted or j in restricted:
                        #continue
                    #cut_sizes.append([i,j, calculate_cut_size(i,j, group1.copy(), group2.copy())])
            #print(len(cut_sizes))
            modified_c_sizes = [init_cut_size - item[2] for item in cut_sizes]
            best_cs_index = modified_c_sizes.index(max(modified_c_sizes))
            #print(best_cut_size)
            best_cut = cut_sizes[best_cs_index]
            #print(best_cut)
            best_cuts.append(best_cut)
            group1.remove(best_cut[0])
            group2.remove(best_cut[1])
            group1.append(best_cut[1])
            group2.append(best_cut[0])
            restricted += 2
            #restricted.append(best_cut[0])
            #restricted.append(best_cut[1])

            group_1_for_pick, group_2_for_pick = groups_bck_up[0], groups_bck_up[1]
            group_1_for_pick.remove(best_cut[0])
            group_2_for_pick.remove(best_cut[1])

        best_cut_index = best_cuts.index(min(best_cuts, key=lambda x: x[2]))
        for i in range(len(best_cuts)-1,best_cut_index,-1):
            item = best_cuts[i]
            group1.remove(item[1])
            group2.remove(item[0])
            group1.append(item[0])
            group2.append(item[1])

        new_cut_size = calculate_cut_size(0, 0, group1, group2, False)
        print(new_cut_size)
    cut_connecting_edges(group1, group2)
    safe_matrix_as_csv()

def cut_connecting_edges(group1, group2):
    for row in group1:
        for col in group2:
            if matrix[row][col] == 1:
                matrix[row][col] = 0
                matrix[col][row] = 0

def calculate_cut_size(v_i, v_j, group1, group2, switch_groups=True):
    if switch_groups:
        group1.remove(v_i)
        group2.remove(v_j)
        group1.append(v_j)
        group2.append(v_i)
    cut_size = 0
    for row in group1:
        for col in group2:
            cut_size+= matrix[row][col]
    return cut_size

def safe_matrix_as_csv():
    edges = []
    print(sum([sum(row) for row in matrix]))
    for index, row in enumerate(matrix):
        for col_index, col in enumerate(row):
            if col == 1:
                if [index, col_index] in edges or [col_index, index] in edges:
                    #print("Yea")
                    continue
                edges.append([index, col_index])
    print(len(edges))
    write_vertices_to_csv(edges,"kerringan-lin.csv")

def write_vertices_to_csv(edges,name):
    with open(name, mode='w+', newline='') as stats_file:
        csv_writer = csv.writer(stats_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in edges:
            csv_writer.writerow(item)


keringhan_lin()