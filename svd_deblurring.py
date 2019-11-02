import numpy as np
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

def save_vector_img(x, size, name):
    img = np.asarray(x.reshape(size, size), dtype=np.uint8)
    img = Image.fromarray(img, mode='L')
    img.save('{}.png'.format(name))

def save_matrix_img(X, name):
    x = np.asarray(X, dtype=np.uint8)
    img = Image.fromarray(x, mode='L')
    img.save('{}.png'.format(name))

def blur_img(X, G):
    x = np.asarray(X.getdata(), dtype=np.float64).reshape((X.size[1], X.size[0]))
    x = np.transpose(x.flatten()[np.newaxis])
    A = np.zeros(shape=(X.size[0]**2,X.size[1]**2))
    g_vector = G.flatten()
    line_size = G.shape[0]
    l_mid = line_size //2
    img_width = X.size[0]

    for i in range(A.shape[0]):
        for j in range(-l_mid,l_mid+1):
            start = i + j * img_width
            for k in range(-l_mid,l_mid+1):
                point = start + k
                if point < 0:
                    point = A.shape[0]+point
                if point >= A.shape[0]:
                    point -= A.shape[0]
                p = ((j+l_mid)*line_size)+k+l_mid
                A[i,point] = g_vector[p]

    y = np.dot(A,x)
    return y, A

def count_inverse(A, sigma):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    cut_index = -1 #index after which values on diagonal are smaller than sigma
    for i in range(S.shape[0]):
        if S[i] < sigma and cut_index==-1:
            cut_index = i
            S[i] = 0
            continue
        if S[i] < sigma:
            print("zeroing")
            S[i] = 0
        else:
            S[i] = 1/S[i]
    if cut_index == -1:
        cut_index = U.shape[0]
    U = np.transpose(U)
    U = U[0:cut_index] # picks only relevant rows
    S = np.diag(S)
    S = S[0:cut_index, 0:cut_index] #Creates matrix without any 0 on diagonal
    V = np.transpose(V)
    V = V[:,0:cut_index] # picks only relevant columns

    return U, S, V

def deblur(y, A, sigma, size):
    U,S,V = count_inverse(A, sigma)
    x = np.dot(U, y)
    x = np.dot(S,x)
    x = np.dot(V,x)
    return x.reshape(size, size)



a = gaussian_blur_matrices[g_blur_index]
X=Image.open('testImg100.png','r')
X=X.convert('L')
size = X.size[1]
sigma = 0.00001

y, A = blur_img(X, a)
save_vector_img(y,size,'blurredImg00001s')
X = deblur(y, A, sigma, size)
save_matrix_img(X, 'clearedImg00001s')
