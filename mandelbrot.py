import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mandelbrot(c, num_of_iterations):
    z = c
    for i in range(num_of_iterations):
        if abs(z)>2:
            return i
        z = z*z + c
    return 0

def generate_mand_colors(num_of_iters, x_bounds, y_bounds, width,height):
    x = np.linspace(x_bounds[0],x_bounds[1],width)
    y = np.linspace(y_bounds[0],y_bounds[1],height)
    colors = []

    for i in range(width):
        row = []
        for j in range(height):
            row.append(mandelbrot(x[i] + 1j*y[j],num_of_iters))
        colors.append(row)
    return colors


def draw_image(color_map,width,height, iterations):
    #fig, ax = plt.subplots(figsize=(10, 10), dpi=720)
    #norm = colors.PowerNorm(gamma)
    #ax.imshow(z.T, cmap=cmap, origin='lower', norm=norm)
    colors = [(66, 30, 15),(25, 7, 26),(9, 1, 47),(4, 4, 73),(0, 7, 100),(12, 44, 138),(24, 82, 177),(57, 125, 209),(134, 181, 229), (211, 236, 248), (241, 233, 191), (248, 201, 95), (255, 170, 0), (204, 128, 0) , (153, 87, 0), (106, 52, 3)]
    #colors = [(0,0,0),(128,0,0), (255,0,0), (255,96,0), (255,192,0), (255,255,0), (255,0,165)]
    color_step = iterations // (len(colors)-2)
    img = Image.new('RGB',(width, height),'black')
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            #print(color_map[i][j])
            """if color_map[i][j] > 0:
                pixels[i,j] = (255,255,255)
            else:
                pixels[i, j] = (0, 0, 0)"""
            pixels[i,j] = colors[color_map[i][j] // color_step]
    img.save("img.bmp")

w = 1000
h = 1000
iterations = 200
draw_image(generate_mand_colors(iterations,[-2,0.5],[-1.25,1.25],w,h),w,h,iterations)