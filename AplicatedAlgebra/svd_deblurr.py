import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img





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


    """for x in range(width):
        for y in range(height):
            sum = 0
            for kernel_index in range(-kernel_half, kernel_half+1):
                sum += big_matrix[width+x, height+y+kernel_index] * gaussian_kernel[kernel_index+kernel_half]
            blurred_img[x,y] = sum
    #horizontal_blur
    for x in range(width):
        for y in range(height):
            sum = 0
            for kernel_index in range(-kernel_half, kernel_half + 1):
                sum += big_matrix[width + x+kernel_index, height + y] * gaussian_kernel[kernel_index + kernel_half]
            blurred_img[x, y] += sum"""
    return blurred_img

"""x=Image.open('022206002.jpg','r')
x=x.convert('L') #makes it greyscale
y=np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))

y = blur_image(y,create_big_matrix(y), np.array([[0.039206, 0.039798, 0.039997, 0.039798, 0.039206],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039997, 0.040601, 0.040804, 0.040601, 0.039997],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039206, 0.039798, 0.039997, 0.039798, 0.039206]]))#test_jpeg(y,640,480)

y=np.asarray(y,dtype=np.uint8) #if values still in range 0-255!
w=Image.fromarray(y,mode='L')
w.save('out3.jpg')"""

#sub_matrix = np.array([[k * l for k in range(9)] for l in range(8)])
#print(np.flip(sub_matrix,0))
#sigma 1  - np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])
#sigma 10 - np.array([0.198005, 0.200995, 0.202001, 0.200995 ,0.198005])
#sigma 1 - np.array([0.27901, 0.44198, 0.27901])

np.array([[0.077847, 0.123317, 0.077847],
[0.123317, 0.195346, 0.123317],
[0.077847, 0.123317, 0.077847]])

#sigma 1
a = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
[0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
[0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])

#sigma 10
a =np.array([[0.039206, 0.039798, 0.039997, 0.039798, 0.039206],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039997, 0.040601, 0.040804, 0.040601, 0.039997],
[0.039798, 0.040399, 0.040601, 0.040399, 0.039798],
[0.039206, 0.039798, 0.039997, 0.039798, 0.039206]])

#X = np.array(data)
U, S, V = np.linalg.svd(a, full_matrices=False)

def inverse(x):
    return 1/x if x != 0 else 0

S = np.vectorize(inverse)(S)
S = np.diag(S)
#S = np.diag(S)
#V = np.transpose(V)
#U = np.transpose(U)
#S = np.linalg.inv(np.diag(S))
"""print(S)
print(V)"""
#tst = np.dot(np.dot(U,S),V)
"""final_mat = np.zeros(shape=(5,5))
print(a)
#print(tst)
print(np.transpose(V[0][np.newaxis]))
for index, sgm in enumerate(S):
        #print(U[:,index])
        #u_t = np.divide(U[index] ,sgm)
        #final_mat = np.add(final_mat, np.dot(u_t,V[index]))
        #print(np.multiply(np.multiply(U[index],sgm),np.transpose(V[index][np.newaxis])))
        #print(np.transpose(V[index][np.newaxis]))
        final_mat += np.multiply(np.divide(V[index], sgm), np.transpose(U[index][np.newaxis]))
        #final_mat += np.multiply(np.multiply(U[index],sgm),np.transpose(V[index][np.newaxis]))#np.dot(np.outer(U[:,index],sgm),V[index])
print(final_mat)"""
#print(S[0])
#print(np.dot(np.dot(U[0],S[0]),np.transpose(V[0])))
#print(np.outer(U[0],np.transpose(V[0])))







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
    """for x in range(-mid, mid+1):
        for y in range(-mid, mid+1):
            x_p = i+x
            y_p = i+y
            if x_p < 0 or x_p >= bound:
                continue
            if y_p < 0 or y_p >= bound:
                continue
            b[x_p,y_p] = a[x+mid,y+mid]"""

    """if a_index == a.size:
        a_index = 0
    b[i,i] = a[a_index]
    a_index+=1"""

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



#blur_base = blur_image(y,create_big_matrix(y),a)

y = np.dot(b,np.transpose(y.flatten()[np.newaxis]))#blur_image(y,create_big_matrix(y), np.linalg.inv(final_mat))#test_jpeg(y,640,480)
#y = y.reshape(100,100)
img=np.asarray(y.reshape(size,size),dtype=np.uint8) #if values still in range 0-255!
w=Image.fromarray(img,mode='L')
w.save('blurred4.jpg')
print("blurred")

#U, S, V = np.linalg.svd(b, full_matrices=False)

#S = np.vectorize(inverse)(S)
"""thresh = 0.00000001
thresh_i = -1
for i in range(S.shape[0]):
    if S[i] < thresh and thresh_i == -1:
        thresh_i = i
    if S[i] < thresh:
        S[i] = 0
    else:
        S[i] = 1/S[i]"""
#U = U[:,0:thresh_i]
#V = V[0:thresh_i]
#S = S[0:thresh_i]
#x = np.dot(np.dot(np.dot(np.transpose(V),np.diag(S)),np.transpose(U)),y)
#x = x.reshape(size,size)
#x=np.asarray(x,dtype=np.uint8) #if values still in range 0-255!
#w=Image.fromarray(x,mode='L')
#w.save('cleared4.jpg')
#print(final_mat)

a = np.array([-1,2,1])
b = np.array([-1,2,1])
c = np.multiply(np.transpose(a[np.newaxis]),b)
#print(np.identity(n=3) - c)
A = np.array([[2,-4,1],[-1,-1,1],[2,-1,-2]])
a = np.array([-1,-1,2])
b = np.array([-1,-1,2])
c = np.multiply(np.transpose(a[np.newaxis]),b)
M = np.identity(n=3) - c
N = np.array([[2,-4,1],[-1,-1,1],[2,-1,-2]])
M = np.array([[1,0,0],[0,0,-1],[0,-1,0]])
N = np.array([[3,-3,-1],[0,0,-1],[0,-3,2]])
#print((M/(1/3)).dot(A))
#print((np.identity(n=3) - (c/3))*3)
#print(np.matmul(M,N))
#c+= copy.deepcopy(c)
#print(c)
m1 = np.array([[1,2],[2,4]])
m2 = (1/np.sqrt(5))*np.array([[1,-2],[2,1]])
m3 = np.array([[1/5,0],[0,0]])
#print(np.dot(m1,np.dot(m2,m3)))
#print(np.dot(np.array([[1,2],[-2,1]]),np.array([[1,-2],[2,1]])))
U, S, V = np.linalg.svd(np.array([[1,2],[2,4]]), full_matrices=False)
#print("{}\n{}\n{}".format(U,S,V))
#D = np.diag(D)

n1 = (1/15)*np.array([[-10,-10,-5],[-10,11,-2],[-5,-2,14]])#(1/3)*np.array([[2,2,1],[2,-1,-2],[1,-2,2]])
n2 = np.array([[2,2,3],[2,3,4],[1,1,2]])
print(np.dot(n1,n2))
#print(np.transpose(np.array([[1,1],[1,-1]])))