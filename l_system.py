import numpy as np
import matplotlib.pyplot as plt
import math

square = "F+F-F-FF+F+F-F" #90
koch = "F+F--F+F" #60
koch_start = "F--F--F"
tree1 = "F[+F]F[-F]F" #20
tree2 = "FF+[+F-F-F]-[-F+F+F]" #20
tree3 = "F[-F]F[+F][F]"
max_depth = 3
#angle_change = np.pi/9
#stack = []

params = [[square,3,np.pi/2,None,False],[koch,4,np.pi/3,koch_start,True],[tree1,4,np.pi/9,None,False],[tree2,4,np.pi/9,None,False],[tree3,4,np.pi/9,None,False]]

def draw_shape_bck(shape, line_length, position, angle, depth, angle_change, stack, max_depth):
    new_pos = [0, 0]
    for ch in shape:
        #print(angle)
        if ch == "+":
            angle += angle_change
        elif ch == "-":
            angle -= angle_change
        elif ch == "[":
            stack.append(position[:])
        elif ch == "]":
            position = stack.pop()[:]
        elif ch == "F":
            if depth < max_depth:
                draw_shape(shape,line_length,position,angle, depth+1, angle_change, stack, max_depth)

                new_pos[0] = position[0] + line_length * np.cos(angle)
                new_pos[1] = position[1] + line_length * np.sin(angle)
                plt.plot([position[0],new_pos[0]],[position[1],new_pos[1]], 'k-', lw=1)
                position[0] = new_pos[0]
                position[1] = new_pos[1]

def draw_shape(shape, line_length, position, angle, depth, angle_change, stack, max_depth, start_shape=None,use_divider=False):
    new_pos = [0, 0]
    if start_shape is None:
      start_shape = shape
    for ch in start_shape:
        #print(angle)
        if ch == "+":
            angle += angle_change
        elif ch == "-":
            angle -= angle_change
        elif ch == "[":
            stack.append(position[:])
        elif ch == "]":
            position = stack.pop()[:]
        elif ch == "F":
            if depth < max_depth:
                draw_shape(shape,line_length,position,angle, depth+1, angle_change, stack, max_depth, None, use_divider)
                if use_divider:
                    divider = 6**(max_depth-depth+1)
                else:
                    divider = 1
                new_pos[0] = position[0] + line_length * np.cos(angle) /divider
                new_pos[1] = position[1] + line_length * np.sin(angle) /divider
                plt.plot([position[0],new_pos[0]],[position[1],new_pos[1]], 'k-', lw=1)
                position[0] = new_pos[0]
                position[1] = new_pos[1]

#draw_shape(square,1,[0,0],0,0,np.pi/2,[],3,None)
#plt.show()
for param in params:
    plt.axis('equal')
    angle_change = param[2]
    stack=[]
    print(param)
    draw_shape(param[0],1,[0,0],0,0,angle_change,stack,param[1],param[3],param[4])#param[]
    plt.show()
    #draw_shape(tree1,1,[0,0],0,0)
    #plt.show()
    #plt.clf()
#plt.show()