from setuptools import setup
import scipy.stats as ss
import numpy as np
import evoapproxlib as eal
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import os
from multiprocessing import Pool, freeze_support
import subprocess

keys = ['PSNR', 'SSIM', 'MSE']

def baseline(pparams_dict, dict_app):
    r2 = subprocess.run([f"./apps/{dict_app['name']}/app", 
                         f"apps/{dict_app['name']}/{dict_app['training']}",
                         f"{pparams_dict['kind']}",
                         f"{pparams_dict['m0']}", f"{pparams_dict['m1']}", f"{pparams_dict['m2']}", f"{pparams_dict['m3']}",
                         f"{pparams_dict['a0']}", f"{pparams_dict['a1']}", f"{pparams_dict['a2']}", f"{pparams_dict['a3']}"], stdout=subprocess.PIPE)
    dict_app["metricerror"] = "SSIM"

def random_vector_norm(seed, size):
    x = np.arange(0, 65536)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 64) - ss.norm.cdf(xL, scale = 64)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    nums = np.random.default_rng(seed).choice(x, size, p = prob)
    return nums

def compute_error(original, approximate):
    psnr_value = []
    ssim_value = []
    mse_noise = []
    
    for i in range(len(original)):
        origin = cv2.imread(original[i])
        approx = cv2.imread(approximate[i])
        if psnr(origin, approx) != float("inf"):
            psnr_value.append(psnr(origin, approx))
        ssim_value.append(ssim(origin, approx, channel_axis=-1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
        mse_noise.append(mse(origin, approx))
        
    if psnr_value: 
        psnr_value = np.mean(psnr_value)
    else: 
        psnr_value = 0
    
    return psnr_value, np.mean(ssim_value), np.mean(mse_noise)
    

def old_gaussian(params, **kwargs):
    input_images = kwargs.get('input')
    size = kwargs.get('size')
    kind = params['kind']

    values = []
    i = size[0]

    for case in input_images:
        if (kind != 'precise' or not os.path.exists('local-image/gaussian_' + kind + str(i) + '.png')):
            filtered_image = gaussian_kernel(params, case)
            filtered_image = filtered_image.astype(np.uint8)
            edge_image = Image.fromarray(filtered_image)
            edge_image.save('local-image/gaussian_' + kind + str(i) + '.png')
        values.append('local-image/gaussian_' + kind + str(i) + '.png')
        i += 1
        
    #print(values)
    return values

def image_run(i):
    
    filtered_image = gaussian_kernel(comps, input_images[i])
    filtered_image = filtered_image.astype(np.uint8)
    edge_image = Image.fromarray(filtered_image)
    edge_image.save('local-image/gaussian_' + kind + str(i+1) + '.png')
    
    return 'local-image/gaussian_' + kind + str(i+1) + '.png'
    

def gaussian_parallel(params, **kwargs):
    global comps
    comps = params
    global input_images
    input_images = kwargs.get('input')
    global size
    size = kwargs.get('size')
    global kind
    kind = params['kind']

    values = []
    
    if (kind != 'precise'):# or not os.path.exists('local-image/sobel_' + kind + str(i) + '.png')):
        with Pool() as pool:
            pool = Pool(processes=8)
            values = pool.map(image_run, range(0,size[1]))
    else:
        values = gaussian(params, **kwargs)

    #print(values)
    return values

def gaussian_kernel(params, input_image):

  [rows, columns] = np.shape(input_image)  # we need to know the shape of the input grayscale image
  filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

  # Now we "sweep" the image in both x and y directions and compute the output
  for i in range(rows):
      for j in range(columns):
          if ((i < 3) or (i > rows - 4) or (j < 3) or (j > columns - 4)):
              filtered_image[i][j] = 0
          else:
              x1 = params['m0'].calc(input_image[i][j], 16)
              x2 = params['m1'].calc(input_image[i - 1][j - 1], 5)
              x3 = params['m1'].calc(input_image[i - 1][j], 5)
              x4 = params['m1'].calc(input_image[i - 1][j + 1], 5)
              x5 = params['m1'].calc(input_image[i][j - 1], 5)
              x6 = params['m1'].calc(input_image[i][j + 1], 5)
              x7 = params['m1'].calc(input_image[i + 1][j - 1], 5)
              x8 = params['m1'].calc(input_image[i + 1][j], 5)
              x9 = params['m1'].calc(input_image[i + 1][j + 1], 5)
              x10 = params['m2'].calc(input_image[i - 2][j - 1], -3)
              x11 = params['m2'].calc(input_image[i - 2][j], -3)
              x12 = params['m2'].calc(input_image[i - 2][j + 1], -3)
              x13 = params['m2'].calc(input_image[i - 1][j - 2], -3)
              x14 = params['m2'].calc(input_image[i - 1][j + 2], -3)
              x15 = params['m2'].calc(input_image[i][j - 2], -3)
              x16 = params['m2'].calc(input_image[i][j + 2], -3)
              x17 = params['m2'].calc(input_image[i + 1][j - 2], -3)
              x18 = params['m2'].calc(input_image[i + 1][j + 2], -3)
              x19 = params['m2'].calc(input_image[i + 2][j - 1], -3)
              x20 = params['m2'].calc(input_image[i + 2][j], -3)
              x21 = params['m2'].calc(input_image[i + 2][j + 1], -3)
              x22 = params['m3'].calc(input_image[i - 2][j - 2], -2)
              x23 = params['m3'].calc(input_image[i - 2][j + 2], -2)
              x24 = params['m3'].calc(input_image[i + 2][j - 2], -2)
              x25 = params['m3'].calc(input_image[i + 2][j + 2], -2)
              
              x26 = params['a0'].calc(x1, x2)
              x27 = params['a0'].calc(x3, x4)
              x28 = params['a0'].calc(x5, x6)
              x29 = params['a0'].calc(x7, x8)
              x30 = params['a0'].calc(x9, x10)
              x31 = params['a1'].calc(x11, x12)
              x32 = params['a1'].calc(x13, x14)
              x33 = params['a1'].calc(x15, x16)
              x34 = params['a1'].calc(x17, x18)
              x35 = params['a1'].calc(x19, x20)
              x36 = params['a1'].calc(x21, x22)
              x37 = params['a2'].calc(x23, x24)
              x38 = params['a2'].calc(x25, x26)
              x39 = params['a3'].calc(x27, x28)
              x40 = params['a3'].calc(x29, x30)
              x41 = params['a3'].calc(x31, x32)
              x42 = params['a3'].calc(x33, x34)
              x43 = params['a3'].calc(x35, x36)
              x44 = params['a3'].calc(x37, x38)
              x45 = params['a3'].calc(x39, x40)
              x46 = params['a3'].calc(x41, x42)
              x47 = params['a3'].calc(x43, x44)
              x48 = params['a3'].calc(x45, x46)
              x49 = params['a3'].calc(x47, x48)
              
              x50 = x49 - input_image[i - 3][j - 1]
              x51 = x50 - input_image[i - 3][j]
              x52 = x51 - input_image[i - 3][j + 1]
              x53 = x52 - input_image[i - 1][j - 3]
              x54 = x53 - input_image[i - 1][j + 3]
              x55 = x54 - input_image[i][j - 3]
              x56 = x55 - input_image[i][j + 3]
              x57 = x56 - input_image[i + 1][j - 3]
              x58 = x57 - input_image[i + 1][j + 3]
              x59 = x58 - input_image[i + 3][j - 1]
              x60 = x59 - input_image[i + 3][j]
              out = x60 - input_image[i + 3][j + 1]
              
              if (out < 0):	out = 0;
              if (out > 255): out = 255 
              filtered_image[i][j] = out

  return filtered_image

def gaussian(params, inputfile):
    if (params['kind'] != 'precise'):
        r2 = subprocess.run(["./apps/gaussian/app", 
                            f"apps/gaussian/{inputfile}",
                            f"{params['kind']}",
                            f"{params['m0']}",
                            f"{params['m1']}",
                            f"{params['m2']}",
                            f"{params['m3']}",
                            f"{params['a0']}",
                            f"{params['a1']}",
                            f"{params['a2']}",
                            f"{params['a3']}"], 
        stdout=subprocess.PIPE)
        res = r2.stdout.decode("utf-8").split(' ')
        res.pop(0)

        return res
    else:
        files = []
        init = 0
        end = 0
        if inputfile == "training.in":
            init = 1
            end = 70
        else:
            init = 66
            end = 95
        for i in range(init, end + 1):
            files.append("local-image/512x512/" + str(i) + ".precise.gauss.ppm")
            
        return files

def get_inputs(size):
    inputs = []

    for case in range(size[0],size[1] + 1):
        image_file = "inputs/image-dataset/gray-512x512/" + str(case) + ".png"
        inputs.append(np.asarray(Image.open(image_file)))

    inputs = {
        'input': inputs,
        'size': size
    }

    return inputs
