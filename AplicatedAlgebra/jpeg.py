import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

quantization_table = np.array([[16,11,10,16,24,40,51,61],
[12,12,14,19,26,58,60,55],
[14,13,16,24,40,57,69,56],
[14,17,22,29,51,87,80,62],
[18,22,37,56,68,109,103,77],
[24,35,55,64,81,104,113,92],
[49,64,78,87,103,121,120,101],
[72,92,95,98,112,100,103,99]])

def divide_into_blocks(image, block_size=8):
    height = len(image)
    width = len(image[0])
    work_img = np.array(copy.deepcopy(image))
    num_w_cycles = width // 8
    num_h_cycles = width // 8
    sub_matrices = []

    for i in range(num_w_cycles):
        for j in range(num_h_cycles):
            start_x = i * 8
            start_y = j * 8
            end_x = start_x + 8 if start_x + 8 < width else width
            end_y = start_y + 8 if start_y + 8 < height else height
            sub_matrix = work_img[start_y:end_y, start_x:end_x]
            """x_dim = sub_matrix.shape[0]
            y_dim = sub_matrix.shape[1]
            if x_dim != 8:
                to_add = sub_matrix[:,x_dim-1]
                for k in range(8-x_dim):
                    np.column_stack([sub_matrix,to_add])
            if y_dim != 8:
                to_add = sub_matrix[y_dim - 1,]
                for k in range(8 - y_dim):
                    np.vstack([sub_matrix, to_add])"""
            if len(sub_matrix) == 0:
                continue
            sub_matrices.append(sub_matrix)
    return sub_matrices


def count_alpha(number):
    return 1/np.sqrt(2) if number == 0 else 1

def dct(sub_matrix, N=8):
    #transform to zero center (Not part of dct but used here for easier code)
    sub_matrix = np.subtract(sub_matrix,128)
    dct = np.zeros(shape=(N,N))
    coeficients = [[count_alpha(i)*count_alpha(j) for i in range(N)] for j in range(N)]
    cosines = [[np.cos(((2*x +1)*y*np.pi)/(2*N)) for y in range(N)] for x in range(N)]
    for i in range(N):
        for j in range(N):
            temp = 0.0
            for x in range(N):
                for y in range(N):
                    #temp+= sub_matrix[x,y]* np.cos(((2*x +1)*i*np.pi)/(16)) * np.cos(((2*y +1)*j*np.pi)/(2*N))
                    temp += cosines[x][i] * cosines[y][j] * sub_matrix[x,y]


            temp *= (1/np.sqrt(2*N))*coeficients[i][j]
            dct[i, j] = temp
    return dct

def quantizate(sub_matrix, N=8):
    B = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            B[i,j] = int(np.round(sub_matrix[i,j]/quantization_table[i,j]))
    return B

def dequantizate(sub_matrix, N=8):
    B = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            B[i, j] = int(np.round(sub_matrix[i, j] * quantization_table[i, j]))
    return B

def inverse_dct(sub_matrix, N=8):
    dct = np.zeros(shape=(N, N))
    coeficients = [[count_alpha(i)*count_alpha(j) for i in range(N)] for j in range(N)]
    cosines = [[np.cos(((2*x +1)*y*np.pi)/(2*N)) for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(N):
            temp = 0.0
            for x in range(N):
                for y in range(N):
                    temp += coeficients[x][y] * cosines[x][i] * cosines[y][j] * sub_matrix[x,y]

            temp *= 1/np.sqrt(2 * N)
            dct[i][j] = int(np.round(temp))
    #print(dct)
    dct = np.add(dct,128)
    return dct

def create_big_matrix(sub_matrices, width, height):
    columns = []
    dim_w = width //8
    dim_h = height//8
    for i in range(dim_w):
        a= sub_matrices[i*dim_h]
        for j in range(dim_h):
            if j == 0:
                continue
            a = np.vstack([a,sub_matrices[i*dim_h+j]])
        columns.append(a)
    big_one = columns[0]
    #np.hstack([big_one,columns[1]])
    for i in range(1,len(columns)):
        big_one = np.hstack([big_one,columns[i]])
    return big_one

def test_jpeg(img,width,height):
    """img = np.array([[52,55,61,66,70,61,64,73],
[63,59,55,90,109,85,69,72],
[62,59,68,113,144,104,66,73],
[63,58,71,122,154,106,70,69],
[67,61,68,104,126,88,68,70],
[79,65,60,70,77,68,58,75],
[85,71,64,59,55,61,65,83],
[87,79,69,68,65,76,78,94]])"""
    #img = np.random.randint(0,255,(16,16))
    #print(img)
    sub_matrices = divide_into_blocks(img)
    #print("Got {} blocks to process... phew.".format(len(sub_matrices)))
    for i in range(len(sub_matrices)):
        sub_matrices[i] = dct(sub_matrices[i])
        #print(sub_matrices[i])
        sub_matrices[i] = quantizate(sub_matrices[i])
        #print(sub_matrices[i])
        #print()
        sub_matrices[i] = dequantizate(sub_matrices[i])
        #print(sub_matrices[i])
        sub_matrices[i] = inverse_dct(sub_matrices[i])
        #print(sub_matrices[i])
    print("reconstructing")
    j_img = create_big_matrix(sub_matrices,width,height)
    print(j_img)
    print("done")
    return j_img


#test_jpeg()
"""sub_matrix = np.array([[k * l for k in range(9)] for l in range(8)])
tst = sub_matrix[1:1 + 4, ]
print(tst)
print(sub_matrix.shape)"""
#a = np.ones(shape=(8,8))
#print(np.subtract(a,128))
#image = img.imread("img.bmp")
#print(image)
#plt.imshow(image)
#plt.show()

#img = Image.new('L',)
#img = img.convert("RGBA")

x=Image.open('services2.jpg','r')
x=x.convert('L') #makes it greyscale
y=np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))

y = test_jpeg(y,640,480)

y=np.asarray(y,dtype=np.uint8) #if values still in range 0-255!
w=Image.fromarray(y,mode='L')
w.save('outt.jpg')

#pixdata = img.load()