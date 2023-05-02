from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
import os
from tqdm import tqdm
from skimage import feature
from skimage.morphology import dilation

#from pyembroidery import *

#Variables
path = "input/"
imgPath = "input/" + os.listdir(path)[0]
base_image = imgPath
thread_count = 1500
nail_count = 240
min_dist = 30
edge = 2
edge_size = 0
sizing = 5

def points_gen(r, n):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = (r - 1) * np.cos(t) + r
    y = (r - 1) * np.sin(t) + r
    return np.c_[x.astype(int), y.astype(int)]

def plotLineLow(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1+1):
        yield (x, y)
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2*dy

def plotLineHigh(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        yield (x, y)
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2*dx

def bresenham_line(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

def is_close(n, center, offset, length):
  lo = (center - offset) % length
  hi = (center + offset) % length
  if lo < hi:
      return lo <= n <= hi
  else:
      return n >= lo or n <= hi

def start_programm(contrast, name):
    #Open Image
    img = Image.open(base_image)
    img = img.resize((500, 500))
    size = img.size

    #Remove Trasperency
    img = img.convert('RGBA')
    background = Image.new('RGBA', size, (255,255,255))
    img = Image.alpha_composite(background, img)

    #Make B&W
    img = img.convert("L")

    #edge magic
    edges = feature.canny(np.array(img), sigma=edge)
    for _ in range(edge_size): edges = dilation(edges)
    edges = white_to_transparency((ImageOps.invert(Image.fromarray((edges * 255).astype(np.uint8)).convert("L"))))

    #Make RGBA
    img = img.convert('RGBA')

    #Add edge
    img = Image.alpha_composite(img, edges)

    #Make B&W Again
    img = img.convert("L")

    #Circle Crop
    mask = Image.new('L', size, 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=1)
    img = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    img.paste(0, mask=mask)

    #Contraster
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    img.show()

    points = points_gen(img.size[0] / 2, nail_count)
    base_point = 0
    final_line = []
    story_point = []

    img = np.asarray(img).copy()
    for _ in tqdm(range(thread_count), desc="Calculating Threads"):
        #get avg lumin
        vals = dict()
        max_num = len(points)
        story_point.append(base_point)
        lines = ( (i, bresenham_line(*points[base_point], *point))
                for i, point in enumerate(points)
                if not is_close(i, base_point, min_dist, max_num))

        for i, line in lines:
            values = [img[value[1]][value[0]] for value in line]
            vals[sum(values)/len(values)] = i

        line = vals[min(vals.keys())]
        final_line.append((tuple(points[base_point]), tuple(points[line])))

        for pixel in bresenham_line(*points[base_point], *points[line]):
            img[pixel[1]][pixel[0]] = 255

        base_point = line

    final_line = [[[i * sizing for i in x] for x in line] for line in final_line]
    size = size[0]*sizing
    output = Image.new('L', (size, size), 255)
    out_draw = ImageDraw.Draw(output)

    f = open(f'text{name}.txt', 'w')
    for line in final_line:
        out_draw.line((*line[0], *line[1]), 0)
    point_name = []
    for element in story_point:
        if (element < 60):
          point_name.append(f"B{61 - (60 - element)}")

        elif (element < 120):
            point_name.append(f"C{61 - (120 - element)}")

        elif (element < 180):
            point_name.append(f"D{61 - (180 - element)}")

        elif (element < 240):
            point_name.append(f"A{61 - (240 - element)}")
    previu_poin = 0
    for element in point_name:
        if previu_poin == 0:
            previu_poin = element
        else:
            f.write(f"{previu_poin} : {element} \n")
            previu_poin = element


    f.close()
    output.show()
    output.save(f"out{name}.jpg")


contrast = 0.4
name = 0
while contrast <= 1.2:
    name =name + 1
    start_programm(contrast, name)
    contrast = contrast + 0.2