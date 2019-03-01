import random
import math
import matplotlib.pyplot as plt

import sys
from tkinter import tix
from tkinter import ttk

import numpy as np
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
            "population": StringVar(value="50"),
            "generations": StringVar(value="100")
        }
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)
        self.show_gui()

    def show_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()
        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Multiobjective optimization", font=("Arial", 44)).pack(anchor='center')


        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Generations", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["generations"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Population", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["population"], font=("Arial", font_size)).pack(side=RIGHT)


        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def start_alg(self):
        multiobjective_optimalization(int(self.alg_params["generations"].get()),
                                      int(self.alg_params["population"].get()))

    def clear_screen(self):
        self.main_frame.destroy()
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)

def generate_solutions(num_of_solutions):
    return [random.uniform(-55.0,55.0) for x in range(num_of_solutions)]

def generate_offspring(parents, constant):
    offspring = []
    num_of_parents = len(parents)
    for i in range(num_of_parents):
        first_p = random.randint(0,num_of_parents-1)
        second_p = random.randint(0,num_of_parents-1)
        if first_p > second_p:
            hlp = first_p
            first_p = second_p
            second_p = hlp
        while first_p == second_p:
            second_p = random.randint(0,num_of_parents-1)
        offspring.append(sum([parents[first_p], parents[second_p]])/2 + constant)
    return offspring

def test_func1(x):
    return -x**2

def test_func2(x):
    return -(x-2)**2

def test_func(x):
    return [test_func1(x), test_func2(x)]

def count_number_of_dominating(solutions, index_to_check):
    chkd_sol = solutions[index_to_check]
    num_of_better = 0
    for index, solution in enumerate(solutions):
        if index == index_to_check:
            continue
        if solution[0] > chkd_sol[0] and solution[1] > chkd_sol[1]:
            num_of_better += 1
    return num_of_better

def count_dominating_list(solutions):
    """
    First step that counts number of solutions that dominate specific solution
    :param solutions:
    :return:
    """
    dominating_list = []
    for index in range(len(solutions)):
        dominating_list.append(count_number_of_dominating(solutions, index))
    return dominating_list

def find_dominated_solutions(solutions, index_to_check):
    chkd_sol = solutions[index_to_check]
    indexes_of_dominated = []
    for index, solution in enumerate(solutions):
        if index == index_to_check:
            continue
        if solution[0] < chkd_sol[0] and solution[1] < chkd_sol[1]:
            indexes_of_dominated.append(index)
    return indexes_of_dominated

def create_set_of_dominating_solutions(solutions):
    """
    Second step that finds all solutions that are dominated by specific solution
    :param solutions:
    :return:
    """
    set_of_dominated = []
    for index in range(len(solutions)):
        set_of_dominated.append(find_dominated_solutions(solutions,index))
    return set_of_dominated

def create_fronts(numbers_of_dominators, set_of_dominated):
    numbers_of_dominators = numbers_of_dominators.copy()
    fronts = []
    while numbers_of_dominators.count(-1) != len(numbers_of_dominators):
        front = []
        for index, dominator in enumerate(numbers_of_dominators):
            if dominator == 0:
                front.append(index)
                for index_of_dominated in set_of_dominated[index]:
                    numbers_of_dominators[index_of_dominated] -= 1
                numbers_of_dominators[index] = -1
        fronts.append(front)
    return fronts

def diversity_preservation(solutions, fronts):
    distances = []
    for front in fronts:
        distances.append(crowding_distance_assignment(solutions, front))
    return distances



def crowding_distance_assignment(solutions, front):
    """

    :param solutions:
    :param front:
    :return: sorted distances of each solution in front in form of (index, distance)
    """
    l = len(front)
    I = [0 for i in range(l)]
    objectives = [[], []]
    for i, index in enumerate(front):
        objectives[0].append([index, test_func1(solutions[index])])
        objectives[1].append([index, test_func2(solutions[index])])
    for objective in objectives:
        objective.sort(key=lambda pair: pair[1])
        # objectives = [[test_func1(solution) for solution in solutions], [test_func2(solution) for solution in solutions]]
    fmax = [max(objectives[0], key=lambda x: x[1]), max(objectives[1], key=lambda x: x[1])]
    fmin = [min(objectives[0], key=lambda x: x[1]), min(objectives[1], key=lambda x: x[1])]

    I[0] = 999999999999
    I[l-1] = 999999999999
    for index, objective in enumerate(objectives):
        for m in objective:
            for i in range(2, l-1):
                #print((I[i+1]*m[1] - I[i-1] * m[1])/(fmax[index] - fmin[index]))
                I[i] = I[i] + (I[i+1]*m[1] - I[i-1] * m[1])/(fmax[index][1] - fmin[index][1])
    distances = []
    for i in range(len(I)):
        distances.append([objectives[0][i][0], I[i]])
    distances.sort(key=lambda pair: pair[1])
    return distances




def multiobjective_optimalization(num_of_generations, num_of_solutions):
    solutions = generate_solutions(num_of_solutions)
    for i in range(num_of_generations):
        offsprings = generate_offspring(solutions, 1)
        solutions.extend(offsprings)
        solutions_counted = [test_func(x) for x in solutions]
        fronts = create_fronts(count_dominating_list(solutions_counted),create_set_of_dominating_solutions(solutions_counted))
        distances = diversity_preservation(solutions, fronts)
        new_solutions = []
        for index in fronts[0]:
            new_solutions.append(solutions[index])
            if len(new_solutions) == num_of_solutions:
                break
        last_front = 1
        last_index = 0
        while len(new_solutions) < num_of_solutions:
            new_solutions.append(solutions[distances[last_front][last_index][0]])
            last_index += 1
            if len(distances[last_front]) == last_index:
                last_front += 1
                last_index = 0
        solutions = new_solutions

    #result_points = [[test_func(x)] for x in solutions]
    plt.xlabel('Function 1 -(x**2)', fontsize=15)
    plt.ylabel('Function 2 -(x-2)**2', fontsize=15)
    plt.scatter([test_func1(x) for x in solutions], [test_func2(x) for x in solutions])
    plt.show()


normal_run = True
if normal_run:
    root = tix.Tk()
    main_menu = Menu(root)
    root.wm_title("Algorithms")
    fc = AlgorithmGUI(root)
    root.config(menu=main_menu)
    root.mainloop()
#multiobjective_optimalization(100, 40)