import numpy as np
import copy

def divide_into_blocks(image, block_size=8):
    height = len(image)
    width = len(image[0])
    work_img = np.array(copy.deepcopy(image))
    num_w_cycles = width//8
    num_h_cycles = width//8
    sub_matrices = []
    miss_x = width - num_w_cycles*8
    miss_y = height - num_h_cycles*8

    for i in range(num_w_cycles):
        for j in range(num_h_cycles):
            start_x = i*8
            start_y = j*8
            end_x = start_x+8 if start_x+8 < width else width
            end_y = start_y+8 if start_y+8 < height else height
            sub_matrix = work_img[start_x:end_x,start_y:end_y]
            if sub_matrix.

            sub_matrices.append(sub_matrix)
 

sub_matrix = np.array([[k*l for k in range(8)] for l in range(8)])
tst = sub_matrix[1:1+4, 1:5]
print(tst)
print(sub_matrix)