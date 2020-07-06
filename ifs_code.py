import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

a1 = np.array([[0.0,0.0,0.01],[0.0,0.26,0.0],[0.0,0.0,0.05]])
jkl1 = np.array([0.0,0.0,0.0])
a2 = np.array([[0.2,-0.26,-0.01],[0.23,0.22,-0.07],[0.07,0.0,0.24]])
jkl2 = np.array([0.0,0.8,0.0])
a3 = np.array([[-0.25,0.28,0.01],[0.26,0.24,-0.07],[0.07,0.0,0.24]])
jkl3 = np.array([0.0,0.22,0.0])
a4 = np.array([[0.85,0.04,-0.01],[-0.04,0.85,-0.09],[0.0,0.08,0.84]])
jkl4 = np.array([0.0,0.8,0.0])
base_matrix = [[a1,jkl1],[a2,jkl2],[a3,jkl3],[a4,jkl4]]

#print(base_matrix[0][0].dot(np.array([0,0,0]))+base_matrix[3][1])

def transform_point(point):
    rnd = np.random.uniform()
    if rnd < 0.01:
        return base_matrix[0][0].dot(point)+base_matrix[0][1]
    elif rnd < 0.08:
        return base_matrix[1][0].dot(point)+base_matrix[1][1]
    elif rnd < 0.15:
        return base_matrix[2][0].dot(point)+base_matrix[2][1]
    else:
        return base_matrix[3][0].dot(point)+base_matrix[3][1]

def do_transformations(num_of_iters):
    point = [0.0,0.0,0.0]
    res_points = []
    for i in range(num_of_iters):
        res_points.append(point)
        point = transform_point(point)
    return res_points

def draw_points(points):
    x,y,z = zip(*points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,c="k",s=0.1)
    plt.show()

draw_points(do_transformations(10000))