import numpy as np
import copy
from PIL import Image

gaussian_blur_matrices = np.array([np.array([[0.077847, 0.123317, 0.077847],
[0.123317, 0.195346, 0.123317],
[0.077847, 0.123317, 0.077847]]),
#sigma 1
np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
[0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
[0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]),

#sigma 10
np.array([[0.039206, 0.039798, 0.039997, 0.039798, 0.039206],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039997, 0.040601, 0.040804, 0.040601, 0.039997],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039206, 0.039798, 0.039997, 0.039798, 0.039206]])])

g_blur_index = 2


def create_big_matrix(base_matrix):
    right_side = copy.deepcopy(base_matrix)
    right_side = np.flip(right_side,1)
    left_side = copy.deepcopy(right_side)
    middle = copy.deepcopy(base_matrix)
    middle = np.hstack([left_side,middle])
    middle = np.hstack([middle,right_side])

    up_side = copy.deepcopy(base_matrix)
    up_side = np.flip(up_side,0)
    up_r_side = np.flip(up_side,1)
    up_l_side = copy.deepcopy(up_r_side)
    up_side = np.hstack([up_l_side,up_side])
    up_side = np.hstack([up_side,up_r_side])

    down_side = copy.deepcopy(up_side)

    big_matrix = np.vstack([up_side,middle])
    big_matrix = np.vstack([big_matrix,down_side])
    return big_matrix

def blur_image(img_matrix,big_matrix, gaussian_kernel):
    kernel_size = gaussian_kernel.shape[0]
    kernel_half = kernel_size // 2
    width = img_matrix.shape[0]
    height = img_matrix.shape[1]
    blurred_img = np.zeros(shape=(width,height))
    #vertical_blur
    for x in range(width):
        for y in range(height):
            s = 0
            for k_i in range(-kernel_half, kernel_half+1):
                for k_j in range(-kernel_half, kernel_half+1):
                    s += big_matrix[x + width + k_i, y + height + k_j] * gaussian_kernel[kernel_half+k_i, kernel_half+k_j]
            blurred_img[x,y] = s
    return blurred_img

def inverse(x):
    return 1/x if x != 0 else 0


a = gaussian_blur_matrices[g_blur_index]
x=Image.open('lena2.png','r')
x=x.convert('L') #makes it greyscale
y=np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
b = np.zeros(shape=(y.shape[0]**2,y.shape[1]**2))

size = y.shape[0]
mid = 5//2
bound = y.shape[0]
num = 5
a *= 2
n_c = y.shape[0]**2//num
for i in range(n_c):#0,y.shape[0],3):
    b[i * num:(i + 1) * num, i * num:(i + 1) * num] = a

for i in range(b.shape[0]):
    for j in range(-mid, mid+1):
        pos = i+j
        if pos > 0 and pos < b.shape[0]:
            if b[i,pos] <=0:
                if pos+1>=b.shape[0]:
                    break
                left = b[i,pos-1]
                right = b[i,pos+1]
                if left > 0:
                    b[i,pos] = left
                elif right > 0:
                    b[i,pos] = right
                else:
                    b[i,pos] = b[i,i]

y = np.dot(b,np.transpose(y.flatten()[np.newaxis]))#blur_image(y,create_big_matrix(y), a)#test_jpeg(y,640,480)
img=np.asarray(y.reshape(size,size),dtype=np.uint8) #if values still in range 0-255!
w=Image.fromarray(img,mode='L')
w.save('blurred4.jpg')

U, S, V = np.linalg.svd(b, full_matrices=False)

thresh = 0.00000001
thresh_i = -1
for i in range(S.shape[0]):
    if S[i] < thresh and thresh_i == -1:
        thresh_i = i
    if S[i] < thresh:
        S[i] = 0
    else:
        S[i] = 1/S[i]

x = np.dot(np.dot(np.dot(np.transpose(V),np.diag(S)),np.transpose(U)),y)
x = x.reshape(size,size)
x=np.asarray(x,dtype=np.uint8) #if values still in range 0-255!
w=Image.fromarray(x,mode='L')
w.save('cleared4.jpg')