from setuptools import setup
import scipy.stats as ss
import numpy as np
import evoapproxlib as eal
import collections
import subprocess

keys = ['MAE', 'WCE', 'MRED', 'MSE', 'EP', 'PROB', 'KEYS', 'BIAS', 'HD']

def compute_error(original, approximate):
    # compute the error distance ED := |a - a'|
    
    error_distance = [abs(approximate[x] - original[x])
        for x in range(0,len(original))]

    square_error_distance = [error_distance[x]**2 for x in range(0,len(error_distance))]

    # compute the relative error distance RED := ED / a
    relative_error_distance = [
        0 if original[x] == 0 else error_distance[x]/original[x]
        for x in range(0,len(original))]
    
    bias_error_distance = [approximate[x] - original[x]
        for x in range(0,len(original))]
    
    #hamming_distance = [hammingDistance(approximate[x], original[x])
    #    for x in range(0,len(original))]
    
    bias_error = [
        0 if original[x] == 0 else bias_error_distance[x]/original[x]
        for x in range(0,len(original))]

    counter = collections.Counter(error_distance)

    total = sum(counter.values())
    keys = list(counter.keys())
    values = list(counter.values())

    pon_avg = 0
    prob = []
    for x in range (0,len(counter.keys())):
        per = round(values[x]/total,6)
        pon_avg += (int(keys[x]) * per)
        prob.append(per)
        
    #plt.bar(keys, prob)
    #plt.show()
    
    # 1 - Normalized Error Distance MED := sum { ED(bj,b) * pj }: MED
    # 2 - Worst Case Error: WCE
    # 3 - Mean Relative Error Distance: MRED
    mred = sum(relative_error_distance)/len(relative_error_distance)
    # 4 - Mean Square Error Distance: MSED
    msed = sum(square_error_distance)/len(square_error_distance)
    # 5 - Error Probability: EP 
    ep = [1 if original[x] != approximate[x] else 0 for x in range(0,len(original))]
    # 6 - p(ED)
    #prob.reverse()
    # 7 - Error Bias (divided by maximum output)
    bias = sum(bias_error)/65535
    return round(pon_avg,3), max(keys), round(mred,3), round(msed,3), round((sum(ep)/len(ep)) * 100,2), prob, keys, round(bias,5), 0#, max(hamming_distance)

def random_vector_norm(seed, size):
    x = np.arange(0, 32767)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 64) - ss.norm.cdf(xL, scale = 64)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    nums = np.random.default_rng(seed).choice(x, size, p = prob)
    return nums

def old_algo(params, **kwargs):
    values = []
    x1 = kwargs.get('x1')
    x2 = kwargs.get('x2')
    x3 = kwargs.get('x3')
    x4 = kwargs.get('x4')
    x8 = kwargs.get('x8')
    for i in range(kwargs.get('size')):
        x5 = params['m0'].calc(x1[i], x2[i])
        x6 = params['a0'].calc(x3[i], x4[i])
        if x6 == 0: x7 = 0
        else: x7 = (x5 / x6)
        x9 = params['m1'].calc(x8[i], x7)
        values.append(x9)
    return values

def algo(params, inputfile):
    r2 = subprocess.run(["./apps/algo1/app", 
                         f"apps/algo1/{inputfile}", 
                         f"{params['m0']}", 
                         f"{params['m1']}", 
                         f"{params['a0']}"], 
    stdout=subprocess.PIPE)
    res = r2.stdout.decode("utf-8").split(' ')
    res.pop(0)
    res = [int(i) for i in res]
    
    return res
    

def get_inputs(size):
    inputs = {
        'x1': random_vector_norm(1234, size),
        'x2': random_vector_norm(5678, size),
        'x3': random_vector_norm(9012, size),
        'x4': random_vector_norm(3456, size),
        'x8': random_vector_norm(7890, size),
        'size': size
    }
    return inputs
