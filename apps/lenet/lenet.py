from setuptools import setup
import scipy.stats as ss
import numpy as np
import evoapproxlib as eal
import collections
import subprocess
import os

keys = ['ED', 'accuracy', 'RED']

def baseline(pparams_dict, dict_app):
    r2 = subprocess.run([f"./apps/lenet/app", f"{dict_app['training']}",
                     f"{pparams_dict['m0']}", f"{pparams_dict['m1']}", f"{pparams_dict['m2']}", f"{pparams_dict['m3']}",
                     f"{pparams_dict['a0']}", f"{pparams_dict['a1']}"], stdout=subprocess.PIPE)
    dict_app["metricerror"] = "accuracy"
    res = r2.stdout.decode("utf-8")
    #res.pop(0)
    #res = [int(i) for i in res]
    
    return float(res)

def compute_error(original, approximate):
    # compute the error distance ED := |a - a'|
    
    return abs(original[0] - approximate[0]), (1.0 - approximate[0]), ((approximate[0] / original[0]) - 1)

def lenet(params, inputfile):
    if (params['kind'] == 'precise'):
        if inputfile == "training":
            return [0.0066]
        else:
            return [0.0160]
    r2 = subprocess.run(["./apps/lenet/app",
                         f"{inputfile}",
                         f"{params['m0']}", 
                         f"{params['m1']}",
                         f"{params['m2']}", 
                         f"{params['m3']}",
                         f"{params['a0']}", 
                         f"{params['a1']}"], 
    stdout=subprocess.PIPE)
    res = r2.stdout.decode("utf-8")#.split(' ')
    #res.pop(0)
    #res = [int(i) for i in res]
    
    return [float(res)]
    
