import copy
import matplotlib.pyplot as plt
import random


def midpoint_displacement(point_s, point_e, r, vert_disp=None, num_of_iterations=10):
    if vert_disp is None:
        vert_disp = (point_s[1] + point_e[1]) / 2

    points = [point_s, point_e]

    for i in range(num_of_iterations):
        points_tmp = copy.deepcopy(points)

        for j in range(len(points_tmp) - 1):

            mid_point = [(points_tmp[j][0] + points_tmp[j + 1][0]) / 2, (points_tmp[j][1] + points_tmp[j + 1][1]) / 2]
            rnd = random.random()

            mid_point[1] += -vert_disp if rnd < 0.5 else vert_disp
            if mid_point[1] < 0:
                mid_point[1] = 0
            points.append(mid_point)
        points = sorted(points)
        vert_disp *= 2 ** (-r)

    return points


def draw_points(points, colors, width, height):
    fig, ax = plt.subplots()
    circle1 = plt.Circle((width - 25, height - 25), 15, color='#ffff00')
    ax.fill_between([0, width], [height, height], color='#9eb2ff')
    ax.add_artist(circle1)
    for index, point_group in enumerate(points):
        x, y = zip(*point_group)
        ax.fill_between(x, y, color=colors[index])


# print(midpoint_displacement([0,0],[20,7],1.2,num_of_iterations=20))
# draw_points(midpoint_displacement([0,0],[400,10],1.2,vert_disp=25,num_of_iterations=16))
# draw_points(midpoint_displacement([0,0],[400,20],1.2,vert_disp=25,num_of_iterations=16))
# draw_points(midpoint_displacement([0,0],[400,50],1.1,num_of_iterations=16))
# draw_points(midpoint_displacement([0,0],[400,25],0.9,vert_disp=25,num_of_iterations=16))
width = 400
a = midpoint_displacement([0, 100], [width, 60], 1.2, vert_disp=25, num_of_iterations=16)
b = midpoint_displacement([0, 60], [width, 40], 1.2, vert_disp=25, num_of_iterations=16)
c = midpoint_displacement([0, 40], [width, 10], 1.1, vert_disp=25, num_of_iterations=16)
d = midpoint_displacement([100, 0], [width, 40], 0.9, vert_disp=10, num_of_iterations=16)
e = midpoint_displacement([0, 10], [300, 0], 1.9, vert_disp=10, num_of_iterations=16)
f = midpoint_displacement([300, 0], [width, 10], 1.9, vert_disp=10, num_of_iterations=16)
colors = ['#ab9481', '#7e5958', '#6f404b', '#724244', '#17981a', '#17981a', '#5d4d48']
draw_points([a, b, c, d, e, f], colors, width, 150)

plt.show()
