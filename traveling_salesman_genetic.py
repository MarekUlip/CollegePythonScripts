import math
import random
import sys
from tkinter import tix
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors as mcolors
from tkinter import *
from functools import partial
import time

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class AlgorithmGUI:
    def __init__(self, gui):
        self.gui = gui
        self.alg_params = {
            "population": StringVar(value="20"),
            "generations": StringVar(value="500"),
            "cross_prob": StringVar(value="0.75"),
            "mutation_prob": StringVar(value="0.5"),
            "alpha": StringVar(value="0.2"),
            "beta": StringVar(value="0.6"),
            "ro": StringVar(value="0.4"),
            "r": StringVar(value="0"),
            "tau": StringVar(value="0.2"),
            "q": StringVar(value="0.5"),

        }
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)
        self.show_gui_ant()

    def show_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()
        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Genetic TSP", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Population", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["population"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Generations", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["generations"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Cross Probability", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["cross_prob"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Mutation Probability", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["mutation_prob"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Draw speed", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["draw_speed"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_gui_ant(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()
        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Ant Colony Optimization", font=("Arial", 44)).pack(anchor='center')


        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Generations", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["generations"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Alpha", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["alpha"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Beta", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["beta"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Ro", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["ro"], font=("Arial", font_size)).pack(side=RIGHT)

        """wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="R", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["r"], font=("Arial", font_size)).pack(side=RIGHT)"""

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Tau", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["tau"], font=("Arial", font_size)).pack(side=RIGHT)

        """wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Q", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["q"], font=("Arial", font_size)).pack(side=RIGHT)"""

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def start_alg(self):
        """genetic_algorithm(int(self.alg_params["population"].get()),
                          int(self.alg_params["generations"].get()),
                          float(self.alg_params["cross_prob"].get()),
                          float(self.alg_params["mutation_prob"].get()),
                          int(self.alg_params["draw_speed"].get()))"""
        ant_colony_optimization(int(self.alg_params["generations"].get()),
                                alpha=float(self.alg_params["alpha"].get()),
                                beta=float(self.alg_params["beta"].get()),
                                ro=float(self.alg_params["ro"].get()),
                                r=float(self.alg_params["r"].get()),
                                base_tau=float(self.alg_params["tau"].get()),
                                q=float(self.alg_params["q"].get()))

    def clear_screen(self):
        self.main_frame.destroy()
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)

def genetic_algorithm(pop_num, generations, cross_prob, mutation_factor, drawing_speed):
    num_of_towns = 20
    population = []
    to_draw = []
    towns = [[60, 200], [80, 200], [80, 180], [140, 180], [20, 160], [100, 160], [200, 160], [140, 140], [40, 120], [100, 120], [180, 100], [60, 80], [120, 80], [180, 60], [20, 40], [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]]
    for i in range(pop_num):
        person = []
        for j in range(num_of_towns):
            rand_num = random.randint(0, 19)
            while rand_num in person:
                rand_num = random.randint(0, 19)
            person.append(rand_num)
        population.append(person)
    distance_matrix = create_distance_matrix(towns)
    #print(distance_matrix)
    for i in range(generations):
        #for index, path in enumerate(population):

            #print("{}".format(index))
        parent = roulette_selection(population,distance_matrix)
        parent2 = roulette_selection(population,distance_matrix)
        """while parent == parent2:
            print("Roulleting")
            parent2 = roulette_selection(population, distance_matrix)"""
        """rnd = random.randint(0,pop_num-1)
        while rnd == index:
            rnd = random.randint(0,pop_num-1)
        parent = population[rnd]"""

        #print("{} done roulette".format(index))
        wannabe_path = do_crossover(parent, parent2, cross_prob)
        #print("{} done crossover".format(index))
        do_mutation(wannabe_path, mutation_factor)
        #print("{} done mutation".format(index))
        worst = find_extreme(population, distance_matrix, towns, False)[0]
        #print("{} found worst".format(index))
        #if count_cost(population[worst], distance_matrix) > count_cost(wannabe_path, distance_matrix):
        population[worst] = wannabe_path
        #print("{} swapped worst".format(index))
        #if count_cost(path, distance_matrix) > count_cost(wannabe_path, distance_matrix):
        #    population[index] = wannabe_path
        """path_costs = {}
        for index, path in enumerate(population):
            path_costs[index] = count_cost(path,distance_matrix)
        best = sorted(path_costs.items(), key=lambda kv: kv[1])[0]
        best_points = []
        for town in population[best[0]]:
            best_points.append(towns[town])"""
        candidate = find_extreme(population,distance_matrix,towns,True)[1]
        if candidate not in to_draw:
            to_draw.append(candidate)
        #print(count_cost(to_draw[i][len(to_draw)-1],distance_matrix))
        print(len(to_draw))
    draw_graph_animated(to_draw,drawing_speed)
    return to_draw[len(to_draw)-1]

def find_extreme(population, distance_matrix, towns, find_best):
    pops = population.copy()
    path_costs = {}
    #for i in range(len(pops)):
    #    pops[i].append(pops[i][0])
    for index, path in enumerate(population):
        path_costs[index] = count_cost(path, distance_matrix)
    best = sorted(path_costs.items(), key=lambda kv: kv[1])
    best_points = []
    if find_best:
        print(best[0])
        #print(population[best[0][0]])
        best = best[0]
    else:
        best = best[len(best)-1]
    for town in population[best[0]]:
        best_points.append(towns[town])
    return [best[0], best_points, best[1]]

def create_distance_matrix(points):
    size = len(points)
    matrix = [[[] for x in range(size)] for y in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = euclidean_distance(points[i], points[j])
    return matrix


def euclidean_distance(vec1, vec2):
    return math.sqrt(sum([(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))]))

def count_cost(path, distance_matrix):
    path_cost = 0
    for i in range(len(path)-1):
        j = i+1
        path_cost += distance_matrix[path[i]][path[j]]
    path_cost += distance_matrix[path[0]][path[len(path)-1]]
    return path_cost

def roulette_selection(paths, distance_matrix):
    path_costs = [count_cost(path, distance_matrix) for path in paths]
    p_costs_dict = {}
    for i in range(len(paths)):
        p_costs_dict[i] = path_costs[i]
    p_costs_dict = sorted(p_costs_dict.items(), key=lambda kv: kv[1])
    S = sum(path_costs)
    rand_S = random.random()#random.randint(0,int(S))
    path_sum = 0
    parent = []
    """for index, path_cost in enumerate(path_costs):
        if path_sum > rand_S:
            parent = paths[index]
            break
        path_sum += path_cost/S"""
    for index, path_cost in p_costs_dict:
        if path_sum >= rand_S:
            parent = paths[index]
            break
        path_sum += path_cost / S
    if len(parent) == 0:
        print("Uups")
        parent = paths[len(paths)-1]
    return parent

def do_crossover(path, parent, prob):
    rnd = random.random()
    if rnd > prob:
        return path
    start = random.randint(0, len(path))
    end = random.randint(0, len(path))
    while start == end:
        end = random.randint(0, len(path))
    if start > end:
        helper = start
        start = end
        end = helper
    p_slice = parent[start:end]
    #res_path = path[0:start] + p_slice + path[end:]

    #return create_set(res_path)
    fixed_paths = create_set_from_parts(path[0:start],path[end:],p_slice,len(path))
    res_path = fixed_paths[0] + p_slice + fixed_paths[1]
    res_p = sorted(res_path)
    if res_p != [x for x in range(20)]:
        print("Lists do not match. Duplicity suspected.")
    return res_path

def create_set(path):
    missing = []
    duplicates = []
    seen = {}
    for index, i in enumerate(path):
        if i not in seen:
            seen[i] = 1
        else:
            seen[i] += 1
            duplicates.append(i)

    for i in range(len(path)):
        if i not in path:
            missing.append(i)

    for index,item in enumerate(path):
        if item in duplicates:
            path[index] = missing.pop() #TODO randomize
            duplicates.remove(item)
    return path

def create_set_from_parts(part1, part2, duplicates, num):
    missing = []
    for i in range(num):
        if i not in part1 and i not in part2 and i not in duplicates:
            missing.append(i)
    for index, item in enumerate(part1):
        if item in duplicates:
            part1[index] = missing.pop()
    for index, item in enumerate(part2):
        if item in duplicates:
            part2[index] = missing.pop()
    return [part1, part2]


def do_mutation(path, mutation_factor):
    mut = random.random()
    if mut < mutation_factor:
        first_i = random.randint(0,len(path)-1)
        second_i = random.randint(0,len(path)-1)
        while first_i == second_i:
            second_i = random.randint(0,len(path)-1)
        helper = path[first_i]
        path[first_i] = path[second_i]
        path[second_i] = helper
    return path

def draw_graph_animated(points, interval=100):
    """
    x = y = np.arange(minimum, maximum, count_density(minimum, maximum))
    X, Y = np.meshgrid(x, y)
    zs = np.array([test_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=0, antialiased=False)"""
    fg = plt.figure(2)
    a = plt.axes()  # fig.add_subplot(111, projection='3d')

    dat = points[len(points) - 1]
    #dat.append(dat[0])
    a.plot([i[0] for i in dat], [j[1] for j in dat], color='k', marker='.', markersize=10)
    a.plot(dat[0][0], dat[0][1], color='r', marker='.', markersize=20)
    a.set_xlabel('X Label')
    a.set_ylabel('Y Label')

    fig = plt.figure(1)
    ax = plt.axes()#fig.add_subplot(111, projection='3d')
    def update(num):
        if num == len(points)-1:
            anim.event_source.stop()
        to_draw = points[num]
        to_draw.append(to_draw[0])
        #print("{}".format(num))
        graph.set_data([i[0] for i in to_draw], [j[1] for j in to_draw])
        return graph,

    data = points[0]
    data.append(data[0])
    graph, = ax.plot([i[0] for i in data], [j[1] for j in data], color='k', marker='.', markersize=10)
    anim = animation.FuncAnimation(fig, update, interval=interval, blit=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    #ax.view_init(90,180)

    #plt.show()

    plt.show()

#r je 0,4 mozna
def ant_colony_optimization(generations,alpha=0.2, beta=0.6, r=0.4, ro=0.4, q=0.0, base_tau=0.2):
    population = []
    to_draw = []
    towns = [[60, 200], [80, 200], [80, 180], [140, 180], [20, 160], [100, 160], [200, 160], [140, 140], [40, 120],
             [100, 120], [180, 100], [60, 80], [120, 80], [180, 60], [20, 40], [100, 40], [200, 40], [20, 20], [60, 20],
             [160, 20]]
    town_count = len(towns)
    distance_matrix = create_distance_matrix(towns)
    best_len = 99999999
    best_path = []
    feromones_best = [[base_tau for y in range(town_count)] for x in range(town_count)]
    feromones = []

    for i in range(town_count):
        feromones.append(feromones_best.copy())
    for generation in range(generations):
        paths = []
        for ant in range(town_count):
            town_ids = [x for x in range(town_count) if x != ant]
            path = [ant]
            distances = []
            #distance_sum = 0
            for i in range(town_count-1):
                nxt_town = pick_next_town(feromones[ant], path[i],town_ids,distance_matrix, alpha, beta)
                path.append(nxt_town)
                town_ids.remove(nxt_town)
                #distance = count_cost([path[i],nxt_town],distance_matrix)
                #distances.append(distance)
                #recount_feromones_locally(path[len(path)-2],nxt_town,base_tau, ro,feromones[ant])
                #disperse_feromones(0.9, feromones[ant])
                #recount_feromones(ro, feromones[i], distance, distance_sum)
                #distance_sum += distance
            #recount_feromones_locally(path[len(path)-1], path[0], base_tau, ro, feromones[ant])
            paths.append(path)
        candidate = find_extreme(paths, distance_matrix, towns, True)
        if candidate[2] < best_len:
            to_draw.append(candidate[1])
            best_len = candidate[2]
            best_path = paths[candidate[0]]
        disperse_feromones(ro,feromones_best)
        #recount_feromones(ro,feromones_best,best_path,best_len)
        recount_feromones_paths(ro,feromones_best,paths,distance_matrix)
        #disperse_feromones(0.3,feromones_best)
        feromones = []
        for i in range(town_count):
            feromones.append(feromones_best.copy())
    print(best_len)
    draw_graph_animated(to_draw,250)


def pick_next_town(feromones, point_of_origin, towns, distance_matrix, alpha, beta):
    probabilities = []
    feromones_sum = sum([(feromones[point_of_origin][town]**alpha) * ((1 / distance_matrix[point_of_origin][town]) **beta) for town in towns])
    for town in towns:#range(len(towns)):
        feromone_str = float(((feromones[point_of_origin][town]**alpha) * ((1/distance_matrix[point_of_origin][town])**beta))) / float(feromones_sum)
        probabilities.append(feromone_str)
    cummulative_probability = []
    cum_sum = 0
    for probability in probabilities:
        cum_sum += probability
        cummulative_probability.append(cum_sum)
    rnd = random.random()
    for index, prob in enumerate(cummulative_probability):
        if rnd <= prob:
            return towns[index]
    #print("Returning last")
    return towns[len(towns)-1]


def recount_feromones_paths(ro, feromones, paths, distance_matrix):
    delta = 0
    edge_list = []
    for path in paths:
        distance_sum = count_cost(path,distance_matrix)
        if distance_sum != 0:
            delta = 1 / distance_sum
        for i in range(len(path)-1):
            for j in range(1,len(path)):
                feromones[path[i]][path[j]] = (1 - ro) * feromones[path[i]][path[j]] + ro * delta
                #edge_list.append([path[i],path[j]])
                #edge_list.append([path[j],path[i]])

        """for r, row in enumerate(feromones):
            for col, column in enumerate(row):
                if [r,col] in edge_list:
                    feromones[r][col] = (1 - ro) * feromones[r][col] + ro * delta
"""
                    #feromones[path[i]][path[j]] = (1 - ro) * feromones[path[i]][path[j]] + ro * delta
                    #feromones[path[j]][path[i]] = (1 - ro) * feromones[path[j]][path[i]] + ro * delta

def recount_feromones(ro, feromones, path, distance_sum):
    delta = 0
    if distance_sum != 0:
        delta = 1 / distance_sum
    for i in range(len(path)-1):
        for j in range(1,len(path)):
            feromones[path[i]][path[j]] = (1 - ro) * feromones[path[i]][path[j]] + ro * delta
            feromones[path[j]][path[i]] = (1 - ro) * feromones[path[j]][path[i]] + ro * delta
    """for r, row in enumerate(feromones):
        for col, column in enumerate(row):
            delta = 0
            if distance_sum != 0:
                delta = 1/distance_sum
            feromones[r][col] = (1-ro)*feromones[r][col] + ro*delta"""

def disperse_feromones(ro, feromones):
    for r, row in enumerate(feromones):
        for col, column in enumerate(row):
            feromones[r][col] *= (1 - ro) * feromones[r][col]

def recount_feromones_locally(i,j,tau0,ro, feromones):
    feromones[i][j] = (1-ro)*feromones[i][j] + ro*tau0
    feromones[j][i] = (1-ro)*feromones[j][i] + ro*tau0



#genetic_algorithm(20,2000,0.2)
#ant_colony_optimization(100)
normal_run = True
if normal_run:
    root = tix.Tk()
    main_menu = Menu(root)
    root.wm_title("Algorithms")
    fc = AlgorithmGUI(root)
    root.config(menu=main_menu)
    root.mainloop()

