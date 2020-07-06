import matplotlib.pyplot as plt
import numpy as np


def get_dragon_params(iterations):
    axiom = "FX"
    X = "X+YF+"
    Y = "-FX-Y"
    angle = np.pi / 2
    return axiom, X, Y, angle, iterations


def get_hilbert_params(iterations):
    axiom = "X"
    X = "-YF+XFX+FY-"
    Y = "+XF-YFY-FX+"
    angle = np.pi / 2
    return axiom, X, Y, angle, iterations


def get_grown_string(base, shape_rules):
    grown_string = ""
    for ch in base:
        if ch == 'X':
            grown_string += shape_rules[0]
        elif ch == 'Y':
            grown_string += shape_rules[1]
        else:
            grown_string += ch
    return grown_string


def create_shape(iterations, shape_base, shape_rules):
    for i in range(iterations):
        shape_base = get_grown_string(shape_base, shape_rules)
        print(shape_base)
    return shape_base


def draw_shape(shape, line_length, position, angle, angle_change):
    new_pos = [0, 0]
    for ch in shape:
        # print(angle)
        if ch == "+":
            angle += angle_change
        elif ch == "-":
            angle -= angle_change
        elif ch == "F":
            new_pos[0] = position[0] + line_length * np.cos(angle)
            new_pos[1] = position[1] + line_length * np.sin(angle)
            plt.plot([position[0], new_pos[0]], [position[1], new_pos[1]], 'k-', lw=1)
            position[0] = new_pos[0]
            position[1] = new_pos[1]


# print(get_dragon_params()[1:3])
# print(create_shape(3,get_dragon_params()[0],get_dragon_params()[1:3]))

params = [get_dragon_params(10), get_hilbert_params(5)]
plt.axis('equal')
for param in params:
    draw_shape(shape=create_shape(param[4], param[0], param[1:3]), line_length=1, position=[0, 0], angle=0,
               angle_change=param[3])
    plt.show()
# plt.axis('equal')
# plt.show()
