# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:51:41 2024

@author: fische_r
"""

import os
import numpy as np
import xarray as xr
import cupy as cp
import cupyx.scipy.ndimage as GPUndimage
from cucim.skimage.morphology import ball as GPUball
from time import sleep
from joblib import delayed, Parallel
import subprocess

num_GPU=5
GPUlimit = 0.5
njobs = 16


def find_free_GPU_memory(gpu_id, limit=GPUlimit, num_GPU=num_GPU):
    free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
    
    while free<limit:
        gpu_id = (gpu_id+1)%num_GPU #for all available GPUs
        sleep(2)
        free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
        
    return gpu_id

toppath = '/mnt/SSD/fische_r/COELY/05_water_GDL_ML'
gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'

samples = ['1', '3II', '3III', '4', '4II', '5', '5II', '6', '7x', '8x']

# git hash
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)   


def timestep_function(im, i):
    gpu_id = i%num_GPU
    gpu_id = find_free_GPU_memory(gpu_id)
    
    with cp.cuda.Device(gpu_id):
        imgpu = cp.array(im)
        imgpu = GPUndimage.median_filter(imgpu, footprint=GPUball(1))
        imgpu = GPUndimage.binary_fill_holes(imgpu)
        im = cp.asnumpy(imgpu)
        del imgpu
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    return im

def label_slug(im, i):
    gpu_id = i%num_GPU
    gpu_id = find_free_GPU_memory(gpu_id)
    
    with cp.cuda.Device(gpu_id):
        imgpu = cp.array(im)
        imgpu = GPUndimage.label(imgpu)
        slug_labels, counts = cp.unique(imgpu[0][:,:150,:], return_counts=True) #150 as height criterion for slug
        rel_labels = slug_labels[counts>20][1:] #only consider larger labels to speed up calculation
        slug_water = cp.zeros(imgpu.shape, dtype=bool)
        for j in rel_labels:
            slug_water[imgpu[0]==j] = True
        im = cp.asnumpy(slug_water)
        del imgpu, slug_water
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    return im
    
    

def sample_function(sample):
    
    datapath = os.path.join(toppath, sample, sample+'water_segmentation.nc')
    
    data = xr.load_dataset(datapath)
    
    Stack = data['segmented'].data
    shp = Stack.shape
    timesteps = np.arange(shp[3])
    water = Stack == 2
    Stack[water] = 1 #temporarily remove the water from the image
    
    results = Parallel(n_jobs=njobs, temp_folder = '/mnt/SSD/fische_r/tmp')(delayed(timestep_function)(water[...,i], i) for i in timesteps)
    water = np.stack(results, axis=-1)
    Stack[water] = 2 #put water back into the image, spatial shift by median filters "eats" into other phases (1 px deep), at this point acceptable
    
    results = Parallel(n_jobs=njobs, temp_folder = '/mnt/SSD/fische_r/tmp')(delayed(label_slug)(water[...,i], i) for i in timesteps)
    water = np.stack(results, axis=-1)
    Stack[water] = 4 #give the value 4 to the water considered to be slug
    
    data['segmented_filtered'] = (['x','y','z', 'timestep'], Stack)
    
    data.attrs['values'] = '0: GDL&MPL, 1: gas, 2: water, 3: other, 4: water slug (in channel)'
    data.attrs['git_sha_segmentation_filtering'] = git_sha
    data.attrs['githash_segmentation_filtering'] = githash
    
    data.to_netcdf(datapath)
    
for sample in samples:
    print(sample)
    sample_function(sample)
