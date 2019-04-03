from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#l1, l2, m1, m2 - muze byt 1

def get_derivative(state, time, l1,l2,m1,m2):
    theta_1, dot_theta_1, theta_2, dot_theta_2 = state
    g = 9.81#23

    d_theta_1 = (m2*g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - m2*np.sin(theta_1 - theta_2)*(l1*(dot_theta_1**2)*np.cos(theta_1 - theta_2)+l2*dot_theta_2**2) - (m1 + m2)*g*np.sin(theta_1))/l1*(m1+m2*np.sin(theta_1 - theta_2)**2)

    d_theta_2 = ((m1+m2) *(l1*(dot_theta_1**2)*np.sin(theta_1 - theta_2) - g*np.sin(theta_2) + g*np.sin(theta_1)*np.cos(theta_1 - theta_2)) + m2*l2*(dot_theta_2**2)*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2)) / l2*(m1+m2*np.sin(theta_1 - theta_2)**2)

    return dot_theta_1, d_theta_1, dot_theta_2, d_theta_2


def start_pendulum():
    t = np.arange(0,20,0.01)
    theta_1 = np.pi#(2*np.pi)/6
    theta_2 = (5*np.pi)/8
    dot_theta_1 = dot_theta_2 =0
    state = [theta_1, dot_theta_1, theta_2, dot_theta_2]
    l1 = l2 = m1 = m2 = 1

    od = odeint(get_derivative, state, t,args=(l1,l2,m1,m2))
    print(t)
    print(od)
    theta_1 = od[:,0]
    theta_2 = od[:,2]

    first_point_pos = []
    second_point_pos = []
    for i in range(len(theta_1)):
        x1 = l1*np.sin(theta_1[i])
        y1 = -l1*np.cos(theta_1[i])
        x2 = l1*np.sin(theta_1[i]) + l2*np.sin(theta_2[i])
        y2 = -l1*np.cos(theta_1[i]) - l2*np.cos(theta_2[i])
        first_point_pos.append([x1,y1])
        second_point_pos.append([x2,y2])
    first_point_pos = np.array(first_point_pos)
    second_point_pos = np.array(second_point_pos)
    print(first_point_pos)
    print(second_point_pos)
    print(first_point_pos[:,0])
    draw_graph_animated2(first_point_pos,second_point_pos)
    #draw_graph(None,second_point_pos[:,0], second_point_pos[:,1])
    #draw_graph(None,first_point_pos[:,0], first_point_pos[:,1])



def draw_graph_animated(minimum, maximum, test_function, points, res, interval=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #x = y = np.arange(minimum, maximum, count_density(minimum, maximum))
    #X, Y = np.meshgrid(x, y)
    #zs = np.array([test_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    #Z = zs.reshape(X.shape)

    #ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=0, antialiased=False)

    def update(num):
        if num == len(points)-1:
            anim.event_source.stop()
        to_draw = points[num]
        graph.set_data([i[0] for i in to_draw], [j[1] for j in to_draw])
        #graph.set_3d_properties([test_function([i[0], i[1]]) for i in to_draw])
        return graph,

    data = points[0]
    graph, = ax.plot([i[0] for i in data], [j[1] for j in data], [test_function([i[0], i[1]]) for i in data],color='k', marker='.', markersize=10, linestyle='')
    anim = animation.FuncAnimation(fig, update, interval=interval, blit=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    #ax.view_init(90,180)

def draw_graph_animated2(points1,points2):
    fig, ax = plt.subplots()
    xfixdata, yfixdata = 14, 8
    xdata, ydata = 5, None
    print(len(points1))
    ln, = plt.plot([], [], 'ro-', animated=True)
    #plt.plot([0], [0], 'bo', ms=13)

    def init():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        return ln,

    def update(frame):
        print(frame)
        x1 = points1[frame][0]
        y1 = points1[frame][1]
        x2 = points2[frame][0]
        y2 = points2[frame][1]
        ln.set_data([0,x1, x2], [0,y1, y2])
        return ln,

    ani = animation.FuncAnimation(fig, update, frames=range(len(points1)),
                        init_func=init, blit=True, interval=17)
    plt.show()

def draw_graph(points,x,y):
    plt.scatter(x,y, s=1)
    #plt.plot(x, y, linewidth=3)
    plt.show()

start_pendulum()