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
# naimplementovat blind a hill
# srovnat na testovac9ch funckich jejich vzkonnost
# vykonnost se testuje
# algoritmy poustest 50

# 5 funkci Hill climbing Blind Search
# D = 2
# D = 10
# D = 30
# Vyzualizovat
# matplotlib
# eckli and rosenbrock
# jedinec( parametry) jako pole f(x) je z souradnice Normalni rozdeleni
# vykreslit povrch a pak zakreslovat body
# vyzkouset kolik iteraci potrebuje hill a kolik blind
# vykreslit funkci a tam zakreslovat jedince

# naimplementovat zihani
# pro kazdou funkci vygenerovat 30 experimentu reseni. vzdy ulozit nejlepsi reseni. vysledky zprumerovat a porovnat, kdy je prumer reseni mensi jestli u hill nebo u zihani
# porovnavani priste ukazat ktery algoritmus byl lepsi. Je potreba stejny pocet ohodnoceni ucelove funkce. V kazdem algoritmu musi byt pocet volani ucelove funkce stejny.
# u dimenze 2 by melo stacit 2000 ohodnoceni.
# Vysledna tabulka
#       Hill climb      Sim Zih
# f1
# f2
# f3
# f4
# f5

#pole balancing problem pomoci hill algoritmu


class AlgorithmGUI:
    def __init__(self, gui):
        self.gui = gui
        self.search_algorithm_menu = Menu(main_menu, tearoff=0)
        self.search_algorithm_menu.add_command(label="Blind Search", command=partial(self.change_algorithm, 0))
        self.search_algorithm_menu.add_command(label="Hill Climbing", command=partial(self.change_algorithm, 1))
        self.search_algorithm_menu.add_command(label="Simulated Annealing", command=partial(self.change_algorithm, 2))
        self.search_algorithm_menu.add_command(label="SOMA", command=partial(self.change_algorithm, 3))
        self.search_algorithm_menu.add_command(label="Particle Swarm", command=partial(self.change_algorithm, 4))
        main_menu.add_cascade(label="Search Algorithms", menu=self.search_algorithm_menu)
        test_algorithm_menu = Menu(main_menu, tearoff=0)
        test_algorithm_menu.add_command(label="Test Algorithms", command=self.show_test_gui)
        main_menu.add_cascade(label="Test Algorithms", menu=test_algorithm_menu)
        self.test_function_names = ["Sphere", "Ackley", "Rosenbrock", "Schwefel", "Styblinski-Tang"]
        self.algorithm_names = ["Blind search", "Hill climbing", "Simulated Annealing", "SOMA", "Particle Swarm"]
        self.alg_params = {
            "minimum_var": StringVar(value="-500"),
            "maximum_var": StringVar(value="500"),
            "dimension": StringVar(value="2"),
            "num_of_cycles": StringVar(value="50"),
            "test_function": StringVar(value=self.test_function_names[0]),
            "num_of_identities": StringVar(value="20"),
            "bias": StringVar(value="10"),
            "num_of_tests": StringVar(value="30"),
            "test_algorithms": [StringVar(value=self.algorithm_names[1]), StringVar(value=self.algorithm_names[2])],
            "population": StringVar(value="10"),
            "migrations": StringVar(value="50"),
            "path_length": StringVar(value="3"),
            "step_size": StringVar(value="0.11"),
            "prt": StringVar(value="0.5"),
            "color_pops": IntVar(value=0)
        }
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)
        self.minimum_var = StringVar(value="-500")
        self.maximum_var = StringVar(value="500")
        self.dimension = StringVar(value="2")
        self.num_of_cycles = StringVar(value="50")
        self.test_function = sphere_function
        self.test_functions = [sphere_function, ackley_function, rosenbrock_function, schwefel_function, styblinski_tang_function]

        self.selected_algorithm = 0# StringVar(value=self.algorithm_names[0])
        self.algorithms_gui = [self.show_blind_search_gui, self.show_hill_climging_gui, self.show_simulated_annealing_gui, self.show_soma_gui, self.show_particle_swarm_gui]
        self.algorithms_gui[self.selected_algorithm]()

    def get_test_function(self):
        return self.test_functions[self.test_function_names.index(self.alg_params["test_function"].get())]

    def change_algorithm(self, index):
        self.selected_algorithm = index
        self.algorithms_gui[index]()

    def get_alg_index_by_name(self, name):
        return self.algorithm_names.index(name)

    def show_blind_search_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()
        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Blind Search", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Cycles", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_cycles"], font=("Arial", font_size)).pack(side=RIGHT)

        """wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Algorithms").pack(side=LEFT)
        OptionMenu(wrapper, self.selected_algorithm, *self.algorithm_names).pack(side=RIGHT)"""

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Test Functions", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_function"], *self.test_function_names).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_hill_climging_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()

        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Hill Climbing", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Cycles", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_cycles"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Identities", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_identities"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Standard deviation", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["bias"], font=("Arial", font_size)).pack(side=RIGHT) # smerodatna odchylka

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Test Functions", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_function"], *self.test_function_names).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_simulated_annealing_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()

        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Simulated Annealing", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Cycles", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_cycles"], font=("Arial", font_size)).pack(side=RIGHT)


        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Standard deviation", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["bias"], font=("Arial", font_size)).pack(
            side=RIGHT)  # smerodatna odchylka

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Test Functions", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_function"], *self.test_function_names).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_soma_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()

        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="SOMA", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Population", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["population"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Migrations", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["migrations"], font=("Arial", font_size)).pack(
            side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Test Functions", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_function"], *self.test_function_names).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Step Size", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["step_size"], font=("Arial", font_size)).pack(
            side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Path Length", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["path_length"], font=("Arial", font_size)).pack(
            side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="PRT", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["prt"], font=("Arial", font_size)).pack(
            side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Checkbutton(wrapper, text="Color Populations", variable=self.alg_params["color_pops"], font=("Arial", font_size)).pack(
            side=LEFT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_particle_swarm_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()

        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Particle Swarm", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Particles", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["population"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Iterations", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["migrations"], font=("Arial", font_size)).pack(
            side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Test Functions", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_function"], *self.test_function_names).pack(side=RIGHT)

        """wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Checkbutton(wrapper, text="Color Populations", variable=self.alg_params["color_pops"], font=("Arial", font_size)).pack(
            side=LEFT)"""

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.start_alg, text="Start", font=("Arial", font_size)).pack()

    def show_test_gui(self, clear_screen=True):
        if clear_screen:
            self.clear_screen()

        font_size = 22

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Algoritmhs testing", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Minimum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["minimum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Maximum", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["maximum_var"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Count of test", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_tests"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Dimension", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["dimension"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Cycles", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["num_of_cycles"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Standard deviation", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.alg_params["bias"], font=("Arial", font_size)).pack(side=RIGHT) # smerodatna odchylka

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Algorithms to test:", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.alg_params["test_algorithms"][1], *self.algorithm_names).pack(side=RIGHT)
        Label(wrapper, text="vs", font=("Arial", font_size)).pack(side=RIGHT)
        OptionMenu(wrapper, self.alg_params["test_algorithms"][0], *self.algorithm_names).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.test_algorithms, text="Start", font=("Arial", font_size)).pack(side=LEFT)
        Button(wrapper, command=self.show_test_gui, text="Clear", font=("Arial", font_size)).pack(side=RIGHT)

    def test_algorithms(self):
        results = [[],[]]
        print("\nTest function\t {}\t {}".format(self.alg_params["test_algorithms"][0].get(), self.alg_params["test_algorithms"][1].get()))
        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        tree = ttk.Treeview(wrapper, columns=('Tested function', self.alg_params["test_algorithms"][0].get(), self.alg_params["test_algorithms"][1].get(),"Winner"))
        tree.heading('#0', text='Tested function')
        tree.heading('#1', text=self.alg_params["test_algorithms"][0].get())
        tree.heading('#2', text=self.alg_params["test_algorithms"][1].get())
        tree.heading('#3', text="Winner")
        tree.column('#1', stretch=YES)
        tree.column('#2', stretch=YES)
        tree.column('#0', stretch=YES)

        for index, test_func in enumerate(self.test_functions):
            for i in range(int(self.alg_params["num_of_tests"].get())):
                """results_annealing.append(simmulated_annealing(float(self.alg_params["minimum_var"].get()),
                          float(self.alg_params["maximum_var"].get()),
                          int(self.alg_params["dimension"].get()),
                          reduce_temperature,
                          int(self.alg_params["num_of_cycles"].get()),
                          int(self.alg_params["bias"].get()),
                          test_func,
                          False))
                results_hill.append(hill_climbing_limited(float(self.alg_params["minimum_var"].get()),
                          float(self.alg_params["maximum_var"].get()),
                          int(self.alg_params["dimension"].get()),
                          int(self.alg_params["num_of_cycles"].get()),
                          int(self.alg_params["bias"].get()),
                          test_func))"""
                start = generate_identity(int(self.alg_params["dimension"].get()), float(self.alg_params["minimum_var"].get()), float(self.alg_params["maximum_var"].get()))
                results[0].append(self.start_alg_for_test(test_func,self.get_alg_index_by_name(self.alg_params["test_algorithms"][0].get()), start))
                results[1].append(self.start_alg_for_test(test_func,self.get_alg_index_by_name(self.alg_params["test_algorithms"][1].get()), start))
            winner = self.alg_params["test_algorithms"][0].get() if test_func(self.count_average(results[0])) < test_func(self.count_average(results[1])) else self.alg_params["test_algorithms"][1].get()
            tree.insert('', 'end', text=self.test_function_names[index], values=(self.count_average(results[0]), self.count_average(results[1]), winner))
            print("{}\t {}\t {}\t".format(self.test_function_names[index],self.count_average(results[0]), self.count_average(results[1])))
            print("{}\t {}\t {}\t".format(self.test_function_names[index],test_func(self.count_average(results[0])), test_func(self.count_average(results[1]))))
        tree.pack(fill=BOTH)

    def count_average(self, results):
        dimension = len(results[0])
        average = [0 for x in range(dimension)]
        for item in results:
            for i in range(dimension):
                average[i] += item[i]
        for i in range(dimension):
            average[i] /= len(results)
        return average

    def clear_screen(self):
        self.main_frame.destroy()
        self.main_frame = Frame(self.gui)
        self.main_frame.pack(fill=BOTH)

    def start_alg(self):
        if self.selected_algorithm == 0:
            blind_search(float(self.alg_params["minimum_var"].get()),
                         float(self.alg_params["maximum_var"].get()),
                         int(self.alg_params["dimension"].get()),
                         int(self.alg_params["num_of_cycles"].get()),
                         self.get_test_function())
        if self.selected_algorithm == 1:
            hill_climbing(float(self.alg_params["minimum_var"].get()),
                          float(self.alg_params["maximum_var"].get()),
                          int(self.alg_params["dimension"].get()),
                          int(self.alg_params["num_of_cycles"].get()),
                          int(self.alg_params["num_of_identities"].get()),
                          int(self.alg_params["bias"].get()),
                          self.get_test_function())
        if self.selected_algorithm == 2:
            simmulated_annealing(float(self.alg_params["minimum_var"].get()),
                                 float(self.alg_params["maximum_var"].get()),
                                 int(self.alg_params["dimension"].get()),
                                 reduce_temperature,
                                 int(self.alg_params["num_of_cycles"].get()),
                                 int(self.alg_params["bias"].get()),
                                 self.get_test_function())
        if self.selected_algorithm == 3:
            soma([float(self.alg_params["minimum_var"].get()), float(self.alg_params["maximum_var"].get())],
                 int(self.alg_params["dimension"].get()),
                 int(self.alg_params["population"].get()),
                 int(self.alg_params["migrations"].get()),
                 float(self.alg_params["step_size"].get()),
                 float(self.alg_params["path_length"].get()),
                 float(self.alg_params["prt"].get()),
                 self.get_test_function(),
                 color_pops=True if self.alg_params["color_pops"].get() == 1 else False)

        if self.selected_algorithm == 4:
            particle_swarm([float(self.alg_params["minimum_var"].get()), float(self.alg_params["maximum_var"].get())],
                           int(self.alg_params["dimension"].get()),
                           int(self.alg_params["population"].get()),
                           int(self.alg_params["migrations"].get()),
                           self.get_test_function())

    def start_alg_for_test(self, test_func, index, start):
        if index == 0:
            return blind_search(float(self.alg_params["minimum_var"].get()),
                         float(self.alg_params["maximum_var"].get()),
                         int(self.alg_params["dimension"].get()),
                         int(self.alg_params["num_of_cycles"].get()),
                         test_func,
                         False)
        if index == 1:
            return hill_climbing_limited(float(self.alg_params["minimum_var"].get()),
                          float(self.alg_params["maximum_var"].get()),
                          int(self.alg_params["dimension"].get()),
                          int(self.alg_params["num_of_cycles"].get()),
                          int(self.alg_params["bias"].get()),
                          test_func)
        if index == 2:
            return simmulated_annealing(float(self.alg_params["minimum_var"].get()),
                                 float(self.alg_params["maximum_var"].get()),
                                 int(self.alg_params["dimension"].get()),
                                 reduce_temperature,
                                 int(self.alg_params["num_of_cycles"].get()),
                                 int(self.alg_params["bias"].get()),
                                 test_func,
                                 False)




def normal_distribution(mean, scale, min_max=None):
    element = np.random.normal(mean, scale)
    if min_max is not None:
        wrong = True
        while wrong:
            wrong = False
            if (element < min_max[0]) or (element > min_max[1]):
                wrong = True
                element = np.random.normal(mean, scale)

    return element
    # return 1/(margin*math.sqrt(2*math.pi))*math.e**(x-middle/margin)

def generate_identity_with_normal(prev, scale, min_max=None):
    res = []
    for x in prev:
        res.append(normal_distribution(x, scale,min_max))
    if len(res) == 0:
        print(res)
    return res


def generate_identity(dim, minimum=None, maximum=None, boundaries=None):
    if boundaries is not None:
        minimum = boundaries[0]
        maximum = boundaries[1]
    else:
        if minimum is None or maximum is None:
            raise ValueError("No minimum or maximum was set")
    res = []
    for i in range(dim):
        res.append(random.randint(minimum, maximum))
    return res


def sphere_function(params):
    # res = 0
    """for i in range(len(params)+1):
        res += params[i]**2"""
    """for x in params:
        res += x**2"""
    return sum(x**2 for x in params)


def rosenbrock_function(params):
    res = 0
    for i in range(len(params)-1):
        res += 100*(params[i+1] - params[i]**2)**2 + (1-params[i])**2
    return res


def ackley_function(params, a=20, b=0.2, c=2*math.pi):
    dim = len(params)
    sub_res = -b*math.sqrt(sum([x**2 for x in params])/dim)
    sub_res_2 = sum(math.cos(x*c) for x in params)/dim
    return -a * math.exp(sub_res) - math.exp(sub_res_2) + a + math.exp(1)


def styblinski_tang_function(params):
    return sum([(x**4 - 16*(x**2) + 5*x) for x in params])/2


def rastrigin_function(params):
    A = 10
    dim = len(params)
    return A*dim + sum([(x**2 - A * math.cos(2*math.pi*x) for x in params)])

def schwefel_function(params):
    dim = len(params)
    return 418.9829*dim - sum((x*math.sin(math.sqrt(math.fabs(x)))) for x in params)


def hill_climbing(minimum, maximum, dim, cycles, num_of_identities, scale, test_func):
    all_res = []
    best = generate_identity(dim, minimum, maximum)
    fitness = test_func(best)
    best_tmp = []
    for i in range(cycles):
        for j in range(num_of_identities):
            tmp = generate_identity_with_normal(best, scale, [minimum, maximum])
            all_res.append(tmp)
            fit_tmp = test_func(tmp)
            if fit_tmp < fitness:
                fitness = fit_tmp
                best_tmp = tmp
        if len(best_tmp) == len(best):
            best = best_tmp
    print(best)
    if dim == 2:
        draw_graph(minimum, maximum, test_func, all_res, best)


def hill_climbing_limited(minimum, maximum, dim, cycles, scale, test_func, start=None):
    all_res = []
    if start is None:
        best = generate_identity(dim, minimum, maximum)
    else:
        best = start
    fitness = test_func(best)
    best_tmp = []
    for i in range(cycles):
        tmp = generate_identity_with_normal(best, scale, [minimum, maximum])
        all_res.append(tmp)
        fit_tmp = test_func(tmp)
        if fit_tmp < fitness:
            fitness = fit_tmp
            best = tmp
    #if dim == 2:
    #    draw_graph(minimum, maximum, test_func, all_res, best)
    return best


def blind_search(minimum, maximum, dim, cycles, test_func, draw_res=True):
    fitness = 99999999999
    all_res = []
    res = []
    for i in range(cycles):
        identity = generate_identity(dim, minimum, maximum)
        all_res.append(identity)
        fitness_p = test_func(identity)
        if fitness_p < fitness:
            fitness = fitness_p
            res = identity
    #print(res)
    if dim == 2 and draw_res:
        draw_graph(minimum, maximum, test_func, all_res, res)
    return res


def count_density(minimum, maximum):
    total = math.fabs(minimum) + math.fabs(maximum)
    res = 0.01
    while total/res > 100:
        res *= 10
    return res

def reduce_temperature(temp, alpha = 0.99):
    return alpha*temp

#zvolit teplotu napr. 2000
# vygenerovat teplotu napr x0 = [3,3]
# pak se to vlozi to funkce
# pak vygenerovat pro x0 jedno x1 pomoci normalniho rozdeleni kde x0 a smerodatna odchylka je napr 0.5
# x1 pouzit na funkcu
# porovnat x0 s x1 pokud jsou lepsi, tak je automaticky prijmout.
# x1 nahradi x0 a teplota se snizi.
# pak se t0 ochladi
# opakovani iterace
# pokud je x1 horsi nez x0 tak vygeneruji nahodne cislo v uniformnim rozdeleni a je li mensi nez e na minus delta f/t ted e na -5/1980, pak jeji prijmeme jinak ne
# je razen do lokalniho prohledavani spolu s hill climbing a tabu search - je to otazka ke zkousce
# zkouska je jen z toho co se implementuje
def simmulated_annealing(tf, t0, dim, temp_reduction, cycles, scale, test_func, draw=True, start=None):
    maximum = t0
    if start is None:
        x0 = generate_identity(dim, tf, t0)
    else:
        x0 = start
    results = [x0]
    fitness0 = test_func(x0)
    for i in range(cycles):
        x1 = generate_identity_with_normal(x0, scale, [tf, maximum])  # scale should be 0.5
        results.append(x1)
        fitness1 = test_func(x1)
        delta_f = fitness1 - fitness0
        if delta_f < 0:
            x0 = x1
            fitness0 = fitness1
        else:
            r = random.random()
            #print(math.e**(-1*(delta_f)/t0))
            if r < math.e**(-1*(delta_f)/t0):
                x0 = x1
                fitness0 = fitness1
        t0 = temp_reduction(t0)
        #if t0 < 0.1:
        #    t0 = 0.1
    #print(x0)
    if dim == 2 and draw:
        draw_graph(tf, maximum,test_func,results,x0)
    return x0


#Make SOMA all to one
#Na zacatku se urcuje nekolik parametru
"""
Path Length = 3.0
Step size = 0.11
PRT = 0.1
Mene jedincu vice migracnich cyklu
Na zacatku se populace vygeneruje nahodne. Nesmi byt sousedne. Fakt nahodne. A uniformne. Je dulezite zachovat diverzitu populace - jedinci se musi trochu lisit.
Po vygenerovani jedincu je ohodnotime.
Potom vybereme leadera - jedinec s nejlepsim fitness.
Potom projdeme vsechny jedince.
Vezmeme jedince a vygenerujeme perturbacni vektor. Vektor ma stejnou dimenzi jako jedinec. Pro kazdou dimenzi vygenerujeme nahodne cislo od 0 do 1. a pokud je to cislo mensi nez perturbacni
parametr, pak tam jde jednicka, jinak tam jde 0. Je treba zajistit, aby alespon jeden parametr byl jedna.
Potom skace k leaderovy. Pocet skoku je path length / step size. Po kazdem skoku vypocita a zapamatuje fitness.
Vzorec je pro kazdy prvek z vektrou - x1ML1 = x1 + (xl-x1)*t*PRTVec, kde t je pocet skoku * delka skoku
Napr pro vektor x1 = 2,0 a leader = 1,1, pri PRT = 0,1 je to 1l
x1ML1 = 2 + (1-2)*0.11*0 = 0
= 0+(1-0)*0.11*1 = 0.11
nejlepsi pozici si zapamatujeme a na konci nahradime, pokud byla nejaka nalezena.
Pri dvourozmernem muze byt prt 0.5
50 generaci
"""

def soma(boundaries, dim, pop_size, migrations, step_size=0.11, path_length=3.0, prt=0.5, test_function=sphere_function, draw=True, color_pops=False):
    population = []
    fitness = []
    start_points = []
    leaders = []
    points_to_draw = [[] for x in range(pop_size)]
    end_points = []
    for i in range(pop_size):
        population.append(generate_identity(dim, boundaries=boundaries))
        start_points.append(population[i].copy())
        fitness.append(test_function(population[i]))

    leader = population[fitness.index(min(fitness))]
    leader_fitness = test_function(leader)
    for i in range(migrations):
        for index, person in enumerate(population):
            if leader == person:
                continue
            ptr_vec = generate_prt_vector(prt, dim)
            tmp = []
            best_tmp = person
            fitness_tmp_best = test_function(person)

            for j in range(int(path_length/step_size)):
                tmp = make_jump(leader, person.copy(), step_size*j, ptr_vec, boundaries)
                points_to_draw[index].append(tmp)
                fitness_tmp = test_function(tmp)
                if fitness_tmp < fitness_tmp_best:
                    best_tmp = tmp
                    fitness_tmp_best = fitness_tmp
            population[index] = best_tmp
            fitness[index] = fitness_tmp_best

        leader = population[fitness.index(min(fitness))]
        leaders.append(leader.copy())
        leader_fitness = test_function(leader)
    end_points = population.copy()
    if dim == 2 and draw:
        draw_graph_soma(boundaries[0], boundaries[1], test_function, start_points, points_to_draw, end_points, leader, leaders, color_pops)
    return leader


def make_jump(leader, person, t, ptr_vec, boundaries):
    for i in range(len(person)):
        person[i] += (leader[i] - person[i])*t*ptr_vec[i]
        if person[i] > boundaries[1]:
            person[i] = boundaries[1]
        if person[i] < boundaries[0]:
            person[i] = boundaries[0]
    return person

def generate_prt_vector(prt, dim):
    prt_vec = []
    for i in range(dim):
        prt_vec.append(1 if random.random() < prt else 0)
    return prt_vec

#Particle swarm s vnitrni vahou
def particle_swarm(boundaries, dim, count_of_particles, iterations, test_function, c1=0.2, c2=0.2):
    particles = []
    particle_speeds =[]
    fitness = []
    personal_best = []
    start_points = []
    points_to_draw = [[] for x in range(iterations)]
    end_points = []
    for i in range(count_of_particles):
        particles.append(generate_identity(dim, boundaries=boundaries))
        personal_best.append(particles[i].copy())
        particle_speeds.append(generate_speed(dim, boundaries=boundaries))
        start_points.append(particles[i].copy())
        fitness.append(test_function(particles[i]))
    global_best = particles[fitness.index(min(fitness))]

    for i in range(iterations):
        for j in range(len(particles)):
            particle_speeds[j] = count_speed(dim, particles[j], particle_speeds[j], personal_best[j],global_best,c1,c2,count_w(j,i+1))
            particles[j] = count_new_pos(dim, particles[j], particle_speeds[j],boundaries)
            points_to_draw[i].append(particles[j])
            fitness[j] = test_function(particles[j])
            p_best = test_function(personal_best[j])
            if fitness[j] < p_best:
                personal_best[j] = particles[j].copy()
            if p_best < test_function(global_best):
                global_best = personal_best[j]

    print(global_best)
    if dim == 2:
        draw_graph_pso(boundaries[0], boundaries[1], test_function, points_to_draw, global_best)
    return global_best


def generate_speed(dim, boundaries):
    max_speed = math.fabs(boundaries[0])+math.fabs(boundaries[1])/20
    return generate_identity(dim,boundaries=[0,max_speed])

def count_w(iteration, migration):
    w_start = 0.9
    w_end = 0.4
    return w_start - ((w_start-w_end)*iteration)/migration

def count_speed(dim, position, prev_speed, personal_best, global_best, c1, c2,w):
    next_speed = []
    for i in range(dim):
        next_speed.append(w*prev_speed[i]+ c1*random.random()*(personal_best[i]-position[i]) + c2*random.random()*(global_best[i]-position[i]))
    return next_speed

def count_new_pos(dim, position, speed, boundaries):
    new_pos = []
    for i in range(dim):
        new_pos.append(position[i] + speed[i])
        #udelat to nahodne
        if new_pos[i] > boundaries[1]:
            new_pos[i] = random.randint(boundaries[0], boundaries[1])
        if new_pos[i] < boundaries[0]:
            new_pos[i] = random.randint(boundaries[0], boundaries[1])
    return new_pos


def draw_graph_soma(minimum, maximum, test_function, start_points, points, end_points, res, leaders, color_pops=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(minimum, maximum, count_density(minimum, maximum))
    X, Y = np.meshgrid(x, y)
    zs = np.array([test_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=0, antialiased=False)
    colors = ['gray', 'rosybrown', 'firebrick', 'sienna', 'sandybrown', 'gold', 'darkkhaki', 'palegreen', 'darkgreen', 'mediumspringgreen', 'lightseagreen', 'royalblue'] #list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())[10:]
    for index, population in enumerate(points):
        for p in population:
            if index%2==1:
                continue
            if color_pops:
                ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], color=colors[index], marker='.', markersize=5, alpha=1)
            else:
                ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], color='k', marker='.', markersize=5,
                        alpha=1)

    for p in start_points:
        ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], markerfacecolor='b', markeredgecolor='b',
                marker='.', markersize=10, alpha=1)

    for p in end_points:
        ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], color='lime',
                marker='.', markersize=10, alpha=1)

    for p in leaders:
        ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], color='k',
                marker='P', markersize=10, alpha=1)

    ax.plot([res[0]], [res[1]], [test_function([res[0], res[1]])], markerfacecolor='r', markeredgecolor='r', marker='.', markersize=10, alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def draw_graph_pso(minimum, maximum, test_function, points, res):
    start = points[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(minimum, maximum, count_density(minimum, maximum))
    X, Y = np.meshgrid(x, y)
    zs = np.array([test_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=0, antialiased=False)
    def update(num):
        if num==len(points)-1:
            anim.event_source.stop()
        to_draw = points[num]
        #ax.plot([i[0] for i in to_draw], [j[1] for j in to_draw], [test_function([i[0], i[1]]) for i in to_draw])
        graph.set_data([i[0] for i in to_draw], [j[1] for j in to_draw])
        graph.set_3d_properties([test_function([i[0], i[1]]) for i in to_draw])
        return graph,

    """for migration in points:
        for p in migration:
            ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], color='k', marker='.', markersize=8, alpha=1)"""
    data = points[0]
    graph, = ax.plot([i[0] for i in data], [j[1] for j in data], [test_function([i[0], i[1]]) for i in data],color='k', marker='.', markersize=10, linestyle='')
    anim = animation.FuncAnimation(fig,update, interval=100,blit=True)

    #ax.plot([res[0]], [res[1]], [test_function([res[0], res[1]])], markerfacecolor='r', markeredgecolor='r', marker='.', markersize=10, alpha=1)
    #ax.plot([start[0]], [start[1]], [test_function([start[0], start[1]])], markerfacecolor='b', markeredgecolor='b', marker='.', markersize=10, alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(90,180)



    plt.show()

#TODO add start point
def draw_graph(minimum, maximum, test_function, points, res):
    start = points[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(minimum, maximum, count_density(minimum, maximum))
    X, Y = np.meshgrid(x, y)
    zs = np.array([test_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=0, antialiased=False)

    for p in points:
        ax.plot([p[0]], [p[1]], [test_function([p[0], p[1]])], markerfacecolor='k', markeredgecolor='k', marker='.', markersize=8, alpha=1)

    ax.plot([res[0]], [res[1]], [test_function([res[0], res[1]])], markerfacecolor='r', markeredgecolor='r', marker='.', markersize=10, alpha=1)
    ax.plot([start[0]], [start[1]], [test_function([start[0], start[1]])], markerfacecolor='b', markeredgecolor='b', marker='.', markersize=10, alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



def test_draw_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #x = np.arange(-2.0, 2.0, 0.05)
    #y = np.arange(-1.0,3.0,0.05)
    x = y = np.arange(-400.0, 400.0, 0.5)
    X, Y = np.meshgrid(x, y)
    zs = np.array([rosenbrock_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    plt.show()

def hill_climbing_vs_annealing():

    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b')

    plt.show()"""


#test_draw_graph()

#blind_search(-50,50,2,500, rosenbrock_function)
#[print(normal_distribution(50,20)) for x in range(10)]
#hill_climbing(-500,500,2,5,20,1,styblinski_tang_function)

root = tix.Tk()
main_menu = Menu(root)
root.wm_title("Algorithms")
fc = AlgorithmGUI(root)
root.config(menu=main_menu)
root.mainloop()

def test_soma(dimension,test_func,num_of_tests):
    results = []
    for x in range(num_of_tests):
        results.append(soma([-500,500],dimension,10,50,test_function=test_func,draw=False))
    [print(test_func(x)) for x in results]#print(sum([test_func(x) for x in results])/num_of_tests) #([print(test_func(x)) for x in results]
    dimension = len(results[0])
    average = [0 for x in range(dimension)]
    for item in results:
        for i in range(dimension):
            average[i] += math.fabs(item[i])
    for i in range(dimension):
        average[i] /= len(results)
    print(average)
#test_soma(2,schwefel_function,5)

#soma([-500,500],5,50,100,test_function=schwefel_function)
#simmulated_annealing(-500,500,2,reduce_temperature,500,100,schwefel_function)
#hill_climbing_limited(-500,500,2,50,1,schwefel_function)


