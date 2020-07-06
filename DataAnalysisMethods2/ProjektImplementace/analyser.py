import os
from tkinter import *
from tkinter import filedialog
from tkinter import tix

from helper_network_methods import get_component_matrix
from network_methods import NetworkMethods


class NetworkAnalyser:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_window.wm_title("Algorithms")
        self.main_frame = Frame(self.main_window)
        self.font_size = 22
        self.output_path = self.load_output_path()
        self.available_algoritms = ["Random", "Barabasi", "Link Selection model", "Copy Model", "Biancony"]
        self.gui_inputs = {"node_count": StringVar(value="2000"),
                           "probability": StringVar(value="0.0009"),
                           "node0_count": StringVar(value="10"),
                           "new_edges_count": StringVar(value="3"),
                           "selected_algorithm": StringVar(value=self.available_algoritms[0]),
                           "network_file_path": StringVar(value=""),
                           "first_node_index": StringVar(value="0"),
                           "sample_size": StringVar(value="15"),
                           "k": StringVar(value="3"),
                           "infection_base": StringVar(value="20"),
                           "infection_prob": StringVar(value="20"),
                           "recovery_rate": StringVar(value="2"),
                           "simulation_name": StringVar(value="Simulation"),
                           "save_network": BooleanVar(value=True),
                           "divider": StringVar(value=",")}
        self.network_methods = NetworkMethods(self.output_path)
        self.create_gui()
        self.show_load_network_page()

    def create_gui(self):
        main_menu = Menu(self.main_window)
        self.main_window.config(menu=main_menu)
        main_menu_options_menu = Menu(main_menu, tearoff=0)
        main_menu_options_menu.add_command(label="Generate Network", command=self.show_generate_network_page)
        main_menu_options_menu.add_command(label="Load Network", command=self.show_load_network_page)
        main_menu_options_menu.add_command(label="Set Output Folder", command=self.set_output_path)
        main_menu_options_menu.add_command(label="Other Analysis Methods",
                                           command=self.show_other_analysis_methods_page)
        main_menu.add_cascade(label="Options", menu=main_menu_options_menu)

    def clear_screen(self):
        self.main_frame.destroy()
        self.main_frame = Frame(self.main_window)
        self.main_frame.pack(fill=BOTH)

    def set_output_path(self):
        folder = filedialog.askdirectory()
        if folder is not None and folder != '':
            with open("settings.txt", "w") as f:
                f.write(folder)

    def load_output_path(self):
        exists = os.path.isfile('settings.txt')
        if exists:
            with open("settings.txt", "r") as f:
                path = f.readline()
                print("Output path is set to {}".format(path))
                return path
        else:
            print("No output path set")
            return ""

    def show_generate_network_page(self, clear_screen=True, font_size=None):
        if clear_screen:
            self.clear_screen()

        if font_size is None:
            font_size = self.font_size

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Generate Network", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Network generating algorithm to be used", font=("Arial", font_size)).pack(side=LEFT)
        OptionMenu(wrapper, self.gui_inputs["selected_algorithm"], *self.available_algoritms).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Number of nodes (All):", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["node_count"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Parameter p (Random, Copy, Biancony):", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["probability"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Initial node count (Barabasi, Link selection, Copy, Biancony): ",
              font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["node0_count"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="New edges count (Barabasi, Biancony):", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["new_edges_count"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Checkbutton(wrapper, text="Save generated network", variable=self.gui_inputs["save_network"],
                    font=("Arial", font_size)).pack(side=LEFT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.analyse_network, text="Analyse network", font=("Arial", font_size)).pack()

    def show_load_network_page(self, clear_screen=True, font_size=None):
        if clear_screen:
            self.clear_screen()

        if font_size is None:
            font_size = self.font_size

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Load network from disk", font=("Arial", 44)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="First node index:", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["first_node_index"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Divider:", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["divider"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Path:", font=("Arial", font_size)).pack(side=LEFT)
        Button(wrapper, command=self.get_path_from_disk, text="Browse", font=("Arial", font_size)).pack(side=RIGHT)
        Entry(wrapper, textvariable=self.gui_inputs["network_file_path"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.analyse_network_from_disk, text="Analyse network",
               font=("Arial", font_size)).pack()

    def show_other_analysis_methods_page(self, clear_screen=True, font_size=None):
        header_font_size = 26
        if clear_screen:
            self.clear_screen()

        if font_size is None:
            font_size = self.font_size

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Random Node Sampling", font=("Arial", header_font_size)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Sample size in %", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["sample_size"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.random_node_sampling, text="Analyse sample", font=("Arial", font_size)).pack()

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="K-core", font=("Arial", header_font_size)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="k", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["k"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.k_core, text="Analyse core", font=("Arial", font_size)).pack()

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Simulate SIR model", font=("Arial", header_font_size)).pack(anchor='center')

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Simulation name", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["simulation_name"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Initial infected count in %", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["infection_base"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Infection probability in %", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["infection_prob"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=X)
        Label(wrapper, text="Time till recovery", font=("Arial", font_size)).pack(side=LEFT)
        Entry(wrapper, textvariable=self.gui_inputs["recovery_rate"], font=("Arial", font_size)).pack(side=RIGHT)

        wrapper = Frame(self.main_frame)
        wrapper.pack(fill=BOTH)
        Button(wrapper, command=self.simulate_sir, text="Simulate SIR", font=("Arial", font_size)).pack()

    def get_path_from_disk(self):
        self.gui_inputs["network_file_path"].set(filedialog.askopenfilename())

    def analyse_network(self):
        selected_alg = self.gui_inputs["selected_algorithm"].get().lower()
        params = []
        n = int(self.gui_inputs["node_count"].get())
        params.append("Node count: {}".format(n))

        if selected_alg == 'random':
            p = float(self.gui_inputs["probability"].get())
            params.append("probabilty: {}".format(p))
            self.network_methods.generate_random_graph(n, p)
        elif selected_alg == 'barabasi':
            m = int(self.gui_inputs["new_edges_count"].get())
            n0 = int(self.gui_inputs["node0_count"].get())
            params.append("New edges count: {}".format(m))
            params.append("Initial node count: {}".format(n0))
            self.network_methods.generate_barabasi_albert_graph(n, m, n0)
        elif selected_alg == 'link selection model':
            n0 = int(self.gui_inputs["node0_count"].get())
            params.append("Initial node count: {}".format(n0))
            self.network_methods.link_selection_model(n, n0)
        elif selected_alg == 'copy model':
            n0 = int(self.gui_inputs["node0_count"].get())
            params.append("Initial node count: {}".format(n0))
            p = float(self.gui_inputs["probability"].get())
            params.append("probabilty: {}".format(p))
            self.network_methods.copy_model(n, p, n0)
        elif selected_alg == 'biancony':
            m = int(self.gui_inputs["new_edges_count"].get())
            n0 = int(self.gui_inputs["node0_count"].get())
            params.append("New edges count: {}".format(m))
            params.append("Initial node count: {}".format(n0))
            p = float(self.gui_inputs["probability"].get())
            params.append("probabilty: {}".format(p))
            self.network_methods.biancony_model(p, m, n0, n)
        if self.gui_inputs["save_network"].get():
            self.network_methods.safe_matrix_as_csv("{}/{}".format(self.output_path, selected_alg))
        self.network_methods.analyse_network(params=params)

    def analyse_network_from_disk(self):
        self.network_methods.load_network_from_csv(self.gui_inputs["network_file_path"].get(),
                                                   int(self.gui_inputs["first_node_index"].get()),
                                                   self.gui_inputs["divider"].get())
        self.network_methods.analyse_network()

    def random_node_sampling(self):
        params = ["Random node sampling:"]
        sample_size = int(self.gui_inputs["sample_size"].get())
        params.append("Sample size: {} \%".format(sample_size))
        nodes = self.network_methods.random_node_sampling(sample_size)[1]
        new_network = get_component_matrix(self.network_methods.matrix, nodes)
        self.network_methods.analyse_network(params=params, network=new_network)

    def k_core(self):
        params = ["K-core:"]
        k = int(self.gui_inputs["k"].get())
        params.append("K: {}".format(k))
        nodes = self.network_methods.k_core(k)
        new_network = get_component_matrix(self.network_methods.matrix, nodes)
        self.network_methods.analyse_network(params=params, network=new_network)

    def simulate_sir(self):
        infection_base = int(self.gui_inputs["infection_base"].get())
        infection_prob = int(self.gui_inputs["infection_prob"].get())
        recovery_rate = int(self.gui_inputs["recovery_rate"].get())
        params = ["SIR simulation:"]
        params.append("Amount of infected nodes: {} \%".format(infection_base))
        params.append("Infection probability: {} \%".format(infection_prob))
        params.append("Recovery rate: {} cycles (if < 0 then SI will be used)".format(recovery_rate))
        self.network_methods.simulate_epidemic(
            "{}/{}".format(self.output_path, self.gui_inputs["simulation_name"].get()), infection_base, infection_prob,
            recovery_rate)


root = tix.Tk()
analyser = NetworkAnalyser(root)
root.mainloop()
