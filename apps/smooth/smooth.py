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
import subprocess

keys = ['PSNR', 'SSIM', 'MSE']

def baseline(pparams_dict, dict_app):
    r2 = subprocess.run([f"./apps/{dict_app['name']}/app", 
                         f"apps/{dict_app['name']}/{dict_app['training']}",
                         f"{pparams_dict['kind']}",
                         f"{pparams_dict['a0']}", f"{pparams_dict['a1']}", f"{pparams_dict['a2']}", 
                         f"{pparams_dict['a3']}", f"{pparams_dict['a4']}", f"{pparams_dict['a5']}"], stdout=subprocess.PIPE)
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
    

def old_smooth(params, **kwargs):
    input_images = kwargs.get('input')
    size = kwargs.get('size')
    kind = params['kind']

    values = []
    i = size[0]

    for case in input_images:
        if (kind != 'precise' or not os.path.exists('local-image/smooth_' + kind + str(i) + '.png')):
            filtered_image = smooth_kernel(params, case)
            filtered_image = filtered_image.astype(np.uint8)
            edge_image = Image.fromarray(filtered_image)
            edge_image.save('local-image/smooth_' + kind + str(i) + '.png')
        values.append('local-image/smooth_' + kind + str(i) + '.png')
        i += 1
        
    #print(values)
    return values

def smooth_kernel(params, input_image):

  [rows, columns] = np.shape(input_image)  # we need to know the shape of the input grayscale image
  filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

  # Now we "sweep" the image in both x and y directions and compute the output
  for i in range(rows):
      for j in range(columns):
          if ((i < 2) or (i > rows - 3) or (j < 2) or (j > columns - 3)):
              filtered_image[i][j] = input_image[i][j]
          else:
              x1 = params['a0'].calc(input_image[i - 2][j - 2], input_image[i - 2][j - 1])
              x2 = params['a0'].calc(input_image[i - 2][j - 0], input_image[i - 2][j + 1])
              x3 = params['a0'].calc(x1, input_image[i - 2][j + 2])
              x4 = params['a0'].calc(x2, x3)
            
              x5 = params['a1'].calc(input_image[i - 1][j - 2], input_image[i - 1][j - 1])
              x6 = params['a1'].calc(input_image[i - 1][j - 0], input_image[i - 1][j + 1])
              x7 = params['a1'].calc(x5, input_image[i - 1][j + 2])
              x8 = params['a1'].calc(x6, x7)
              
              x9 = params['a2'].calc(input_image[i - 0][j - 2], input_image[i - 0][j - 1])
              x10 = params['a2'].calc(input_image[i - 0][j - 0], input_image[i - 0][j + 1])
              x11 = params['a2'].calc(x9, input_image[i - 0][j + 2])
              x12 = params['a2'].calc(x10, x11)
              
              x13 = params['a3'].calc(input_image[i + 1][j - 2], input_image[i + 1][j - 1])
              x14 = params['a3'].calc(input_image[i + 1][j - 0], input_image[i + 1][j + 1])
              x15 = params['a3'].calc(x13, input_image[i + 1][j + 2])
              x16 = params['a3'].calc(x14, x15)
              
              x17 = params['a4'].calc(input_image[i + 2][j - 2], input_image[i + 2][j - 1])
              x18 = params['a4'].calc(input_image[i + 2][j - 0], input_image[i + 2][j + 1])
              x19 = params['a4'].calc(x17, input_image[i + 2][j + 2])
              x20 = params['a4'].calc(x18, x19)
              
              x21 = params['a5'].calc(x4, x8)
              x22 = params['a5'].calc(x12, x16)
              x23 = params['a5'].calc(x21, x22)
              out = params['a5'].calc(x23, x20)
              
              out = out / 25
              
              if (out < 0):	out = 0;
              if (out > 255): out = 255 
              filtered_image[i][j] = out

  return filtered_image

def smooth(params, inputfile):
    if (params['kind'] != 'precise'):
        r2 = subprocess.run(["./apps/smooth/app", 
                            f"apps/smooth/{inputfile}",
                            f"{params['kind']}",
                            f"{params['a0']}",
                            f"{params['a1']}",
                            f"{params['a2']}",
                            f"{params['a3']}",
                            f"{params['a4']}",
                            f"{params['a5']}"], 
        stdout=subprocess.PIPE)
        res = r2.stdout.decode("utf-8").split(' ')
        res.pop(0)
        #res = [int(i) for i in res]
        
        return res
    else:
        files = []
        init = 0
        end = 0
        if inputfile == "training.in":
            init = 1
            end = 70
        else:
            init = 71
            end = 100
        for i in range(init, end + 1):
            files.append("local-image/512x512/" + str(i) + ".precise.smoo.ppm")
            
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
