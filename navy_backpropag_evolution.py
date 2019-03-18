import math
import numpy as np


def sigm(x):
    return 1/(1+np.exp(-x))

def count_error(expected, predicted):
    return expected - predicted

def dot_product(input, bias):
    # print(sum([item[0]*item[1] + bias for item in input]))
    return sum([item[0] * item[1] + bias for item in input])

def predict(point, weightsH, weightsO, bias=(1,1)):
    inpts = [[[point[0],weightsH[0]], [point[1],weightsH[1]]], [[point[0],weightsH[2]], [point[1],weightsH[3]]]]
    outsH = []
    outsH.append(sigm(dot_product(inpts[0], bias[0])))
    outsH.append(sigm(dot_product(inpts[1],bias[0])))
    inpts = [[outsH[0],weightsO[0]], [outsH[1],weightsO[1]]]
    out = sigm(dot_product(inpts,bias[1]))
    return outsH, out

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

def generate_identity(dim, minimum=None, maximum=None, boundaries=None):
    if boundaries is not None:
        minimum = boundaries[0]
        maximum = boundaries[1]
    else:
        if minimum is None or maximum is None:
            raise ValueError("No minimum or maximum was set")
    res = []
    for i in range(dim):
        res.append(np.random.uniform(minimum, maximum))
    return res

def generate_identity_with_normal(prev, scale, min_max=None):
    res = []
    for x in prev:
        res.append(normal_distribution(x, scale,min_max))
    if len(res) == 0:
        print(res)
    return res

def hill_climbing(minimum, maximum, dim, cycles, num_of_identities, scale, test_func):
    all_res = []
    best = generate_identity(dim, boundaries=[minimum,maximum])
    fitness = test_func(best)
    best_tmp = []
    for i in range(cycles):
        for j in range(num_of_identities):
            tmp = generate_identity_with_normal(best, scale, [minimum, maximum])
            all_res.append(tmp)
            fit_tmp = test_func(tmp)
            if fit_tmp < fitness:
                print(fit_tmp)
                fitness = fit_tmp
                best_tmp = tmp
        if len(best_tmp) == len(best):
            best = best_tmp
    print(best)
    return best

def test_function(weights):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    error_sum = 0
    for index, point in enumerate(inputs):
        _, out = predict(point,weights[:4], weights[4:6],weights[6:])
        error = count_error(xored[index], out)
        error_sum += math.fabs(error)
    return error_sum


def particle_swarm(boundaries, dim, count_of_particles, iterations, test_function, c1=0.2, c2=0.2):
    particles = []
    particle_speeds =[]
    fitness = []
    personal_best = []
    points_to_draw = [[] for x in range(iterations)]
    for i in range(count_of_particles):
        particles.append(generate_identity(dim, boundaries=boundaries))
        personal_best.append(particles[i].copy())
        particle_speeds.append(generate_speed(dim, boundaries=boundaries))
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
                #print(fitness[j])
                personal_best[j] = particles[j].copy()
            if p_best < test_function(global_best):
                global_best = personal_best[j]

    print(global_best)
    return global_best


def generate_speed(dim, boundaries):
    max_speed = (math.fabs(boundaries[0])+math.fabs(boundaries[1]))/20
    return generate_identity(dim,boundaries=[0,max_speed])


def count_w(iteration, migration):
    w_start = 0.9
    w_end = 0.4
    return w_start - ((w_start-w_end)*iteration)/migration

#TODO pokud rychlost prekroci vMax vygenerovat pro danou dimenzi nahodne novou
def count_speed(dim, position, prev_speed, personal_best, global_best, c1, c2,w):
    next_speed = []
    for i in range(dim):
        next_speed.append(w*prev_speed[i]+ c1*np.random.uniform()*(personal_best[i]-position[i]) + c2*np.random.uniform()*(global_best[i]-position[i]))
    return next_speed

def count_new_pos(dim, position, speed, boundaries):
    new_pos = []
    for i in range(dim):
        value = position[i] + speed[i]
        value = check_boundaries(boundaries,value)
        new_pos.append(value)
    return new_pos

def check_boundaries(boundaries, value):
    if value < boundaries[0] or value > boundaries[1]:
        value = np.random.uniform(boundaries[0], boundaries[1])
    return value

def start_prog():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xored = [0, 1, 1, 0]
    weights = particle_swarm([-30,30],8,100,100,test_function)#hill_climbing(-30.0,30.0,8,500,500,8.0, test_function)


    for index, point in enumerate(inputs):
        #print(predict(point, biases, weightsH, weightsO))
        print("Point {} is expeted to be {} and was predicted as {}".format(point, xored[index], predict(point,weights[:4], weights[4:6],weights[6:])[1]))

start_prog()