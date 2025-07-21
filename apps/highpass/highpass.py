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
                         f"{pparams_dict['m0']}", 
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
    

def old_highpass(params, **kwargs):
    input_images = kwargs.get('input')
    size = kwargs.get('size')
    kind = params['kind']

    values = []
    i = size[0]

    for case in input_images:
        if (kind != 'precise' or not os.path.exists('local-image/highpass_' + kind + str(i) + '.png')):
            filtered_image = highpass_kernel(params, case)
            filtered_image = filtered_image.astype(np.uint8)
            edge_image = Image.fromarray(filtered_image)
            edge_image.save('local-image/highpass_' + kind + str(i) + '.png')
        values.append('local-image/highpass_' + kind + str(i) + '.png')
        i += 1
        
    #print(values)
    return values

def image_run(i):
    
    filtered_image = highpass_kernel(comps, input_images[i])
    filtered_image = filtered_image.astype(np.uint8)
    edge_image = Image.fromarray(filtered_image)
    edge_image.save('local-image/highpass_' + kind + str(i+1) + '.png')
    
    return 'local-image/highpass_' + kind + str(i+1) + '.png'
    

def highpass_parallel(params, **kwargs):
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
        values = highpass(params, **kwargs)

    #print(values)
    return values

def highpass_kernel(params, input_image):

  [rows, columns] = np.shape(input_image)  # we need to know the shape of the input grayscale image
  filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

  # Now we "sweep" the image in both x and y directions and compute the output
  for i in range(rows):
      for j in range(columns):
          if ((i < 1) or (i > rows - 2) or (j < 1) or (j > columns - 2)):
              filtered_image[i][j] = 0
          else:
              x1 = params['m0'].calc(input_image[i][j], 5)
              x2 = -params['a0'].calc(-x1, input_image[i - 1][j])
              x3 = -params['a1'].calc(-x2, input_image[i][j - 1])
              x4 = -params['a2'].calc(-x3, input_image[i][j + 1])
              out = -params['a3'].calc(-x4, input_image[i + 1][j])
              
              if (out < 0):	out = 0
              if (out > 255): out = 255 
              filtered_image[i][j] = out

  return filtered_image

def highpass(params, inputfile):
    if (params['kind'] != 'precise'):
        r2 = subprocess.run(["./apps/highpass/app", 
                            f"apps/highpass/{inputfile}",
                            f"{params['kind']}",
                            f"{params['m0']}",
                            f"{params['a0']}",
                            f"{params['a1']}",
                            f"{params['a2']}",
                            f"{params['a3']}"], 
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
            files.append("local-image/512x512/" + str(i) + ".precise.high.ppm")
            
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
