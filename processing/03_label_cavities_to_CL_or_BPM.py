# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:29:56 2023

@author: fische_r
"""

import os
import skimage
import numpy as np
import warnings
import cupy as cp
import cucim
import scipy as sp
import cupyx.scipy.ndimage as GPUndimage
import xarray as xr
import dask
import dask.array
from joblib import Parallel, delayed
GPUball = cucim.skimage.morphology.ball
ball = skimage.morphology.ball
import socket
host = socket.gethostname()   

# paths
if host == 'mpc2959.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    toppath = '/mpc/homes/fische_r/NAS/DASCOELY'
    toppathSSD = '/mnt/SSD/fische_r/COELY'

elif host == 'mpc2053.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    toppath = '/mpc/homes/fische_r/NAS/DASCOELY'
    toppathSSD = '/mpc/homes/fische_r/NAS/DASCOELY/processing'
else:
    print(host+' is not known to the notebook')
    
segmentation_path = os.path.join(toppath, 'processing', '04_membrane_ML')
registered_path = os.path.join(toppathSSD, '02_registered_3p1D')

px = 2.75e-6
vx = px**3

# fetch githash
import subprocess
cwd = os.getcwd()
os.chdir(gitpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)


th = 15000
thmax = 3*th 


def clean_binary_image(im, minsize = 20, fp_radius = 1, i=0, GPU = True):
    # GPU?
    if GPU:
        warnings.filterwarnings("ignore") #ignore numpy whining and spamming nan-warnings, inside function because joblib spawns workers with default settings on warnings?!   
        gpu_id = i%5 # i%4+1 uses gpus 1 through 4, leaving the big A40 (0) alone or i%5 to use all 5

        with cp.cuda.Device(gpu_id):
            im = cp.array(im)
            im = cucim.skimage.morphology.binary_opening(im, footprint=cucim.skimage.morphology.ball(fp_radius))
            im = cucim.skimage.morphology.remove_small_objects(im, min_size=minsize)
            im = cp.asnumpy(im)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        
    else: 
        im = sp.ndimage.binary_opening(im, structure=skimage.morphology.ball(fp_radius))
        im = skimage.morphology.remove_small_objects(im, min_size=minsize)
        
    return im

def Gaussian_Blur_space(da, sigma):      
    deptharray = np.ones(da.ndim)+4*sigma
    deptharray[-1] = 0
    sigmas = np.ones(deptharray.shape)*sigma
    deptharray = tuple(np.min([deptharray, da.shape], axis=0))

    sigmas[-1] = 0
    G = da.map_overlap(skimage.filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigmas, preserve_range=True)
    return G

def label_gas_cavities(im, imCL, GPU=True, i=0):
    warnings.filterwarnings("ignore") #ignore numpy whining and spamming nan-warnings, inside function because joblib spawns workers with default settings on warnings?!  
    
    im = clean_binary_image(im, i=i, GPU=GPU, minsize=20)
    
    if GPU:
        gpu_id = i%5 # i%4+1 uses gpus 1 through 4, leaving the big A40 (0) alone or i%5 to use all 5

        with cp.cuda.Device(gpu_id):
            im = cp.array(im)
            imCL = cp.array(imCL)
            imCL = GPUndimage.binary_dilation(imCL, GPUball(3))
            im = cucim.skimage.measure.label(im)
            labels, counts = cp.unique(im, return_counts=True)
            CLlabels = cp.unique(im[imCL])
            im = cp.asnumpy(im)
            labels = cp.asnumpy(labels)
            counts = cp.asnumpy(counts)
            CLlabels = cp.asnumpy(CLlabels)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        
    else:
        imCL = sp.ndimage.binary_dilation(imCL, ball(3))
        im = skimage.measure.label(im)
        labels, counts = np.unique(im, return_counts=True)
        CLlabels = np.unique(im[imCL])
        
    return im, labels, counts, CLlabels

def mask_from_membrane(im, minsize = 200, fp_radius = 1, i=0, GPU = True):
    # GPU?
    if GPU:
        warnings.filterwarnings("ignore") #ignore numpy whining and spamming nan-warnings, inside function because joblib spawns workers with default settings on warnings?!   
        gpu_id = i%5 # i%4+1 uses gpus 1 through 4, leaving the big A40 (0) alone or i%5 to use all 5

        with cp.cuda.Device(gpu_id):
            im = cp.array(im)
            im = cucim.skimage.morphology.binary_opening(im, footprint=cucim.skimage.morphology.ball(fp_radius))
            im = cucim.skimage.morphology.remove_small_objects(im, min_size=minsize)
        # find biggest onject and then dilate this
            im = cucim.skimage.measure.label(im)
            labels, counts = cp.unique(im, return_counts=True)
            biggest_label = labels[1:][np.argmax(counts[1:])]
            im = im == biggest_label
            im = cucim.skimage.morphology.binary_dilation(im, footprint=cucim.skimage.morphology.ball(3))
            im = cp.asnumpy(im)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        
    else: 
        im = sp.ndimage.binary_opening(im, structure=skimage.morphology.ball(fp_radius))
        im = skimage.morphology.remove_small_objects(im, min_size=minsize)
        im = skimage.measure.label(im)
        labels, counts = np.unique(im, return_counts=True)
        biggest_label = labels[1:][np.argmax(counts[1:])]
        im = im == biggest_label
        im = sp.ndimage.binary_dilation(im, structure=skimage.morphology.ball(3))
        
    return im

def label_per_time_step(ts, CLim, im, memim):
    
    memim = mask_from_membrane(memim, i=ts)
    
    im = np.bitwise_and(im, memim)
    
    label, label_ids, label_counts, CL_label_ids = label_gas_cavities(im, CLim, i=ts, GPU=False)
    totcav = label_counts[1:].sum()
    CLcav = label_counts[CL_label_ids-1].sum()
    BPMcav = totcav-CLcav
    
    return label, label_ids, label_counts, CL_label_ids, totcav, CLcav, BPMcav, memim


def area_per_position(result):
    label, label_ids, label_counts, CL_label_ids, totcav, CLcav, BPMcav, memim = result
    test = np.zeros(label.shape, dtype=np.uint8)
    for i in label_ids[1:]:
        if i in CL_label_ids:
            col = 2
        else:
            col = 1
        test[np.where(label==i)] = col
        
    return test, memim


segcropdict = {
    '3II': (20,-20,20,-20,250,1600),
    '5': (10,-10,10,-10,10,-10),
    '5II': (10,-10,10,-10,10,-10),
    '1': (10,-10,10,-10,10,-10),
    '4': (10,-10,10,-10,10,-10),
    '3III': (10,-10,10,-10,10,-10),
    '4II': (10,-10,10,-10,200,-10),
    '6': (10,-10,10,-10,10,-10),
    }    

def assign_labels(sample, segcropdict): 

    # TODO: use slightly dilated membrane as mask
    print(sample)
    
    segpath = os.path.join(segmentation_path, sample, sample+'membrane_segmentation.nc')
    seg_data = xr.load_dataset(segpath)
    shp = seg_data['segmented'].shape
    
    [ar,br,cr,dr,er,fr] = seg_data.attrs['05_ML_cropping']
       
    regpath = os.path.join(registered_path, '02_'+sample+'_registered_3p1D.nc')
    reg_data = xr.open_dataset(regpath)
    [ar,br,cr,dr,er,fr] = seg_data.attrs['05_ML_cropping']
    images = [im for im in reg_data.keys() if im[3:7] == 'imag']
    images.sort()
    
    rawim = np.zeros(shp, dtype = np.uint16)
    for i in range(shp[-1]):
        if i%10==0:
            print(i)
        rawim[...,i] = reg_data[images[i]][ar:br,cr:dr,er:fr].data
    
    reg_data.close()
    
    # TODO: adjust cropping for all samples
    a1,b1,c1,d1,e1,f1 = segcropdict[sample]
    
    da = dask.array.from_array(rawim[a1:b1,c1:d1,e1:f1])
    
    da[da>thmax] = 0
    rawim = Gaussian_Blur_space(da, 1)
    rawim = da.compute()
    CLim4D = rawim>th
    
    del rawim
    del da
    
    results = Parallel(n_jobs = 20)(delayed(label_per_time_step)(ts, CLim4D[...,ts], seg_data['segmented'].sel(timestep = ts)[a1:b1,c1:d1,e1:f1].data==2, seg_data['segmented'].sel(timestep = ts)[a1:b1,c1:d1,e1:f1].data==0) for ts in range(shp[-1])) #
    
    label4D = np.zeros(CLim4D.shape, dtype=np.uint16)
    
    cav_vol = np.zeros(shp[-1])
    CL_cav = np.zeros(shp[-1])
    
    for i in range(shp[-1]):
        label4D[...,i] = results[i][0]
        cav_vol[i] = results[i][2][1:].sum()
        CL_cav[i] = results[i][2][1:][results[i][3][1:]-1].sum()
    BPM_cav = cav_vol-CL_cav
    
    shp4D = results[0][0].shape+cav_vol.shape
    

    label_2 = np.zeros(shp4D, dtype=np.uint8)
    mem_mask = np.zeros(shp4D, dtype=np.uint8)
    for ts in range(shp4D[-1]):
         ims = area_per_position(results[ts])
         label_2[...,ts] = ims[0]
         mem_mask[...,ts] = ims[1].astype(np.uint8)
        
    seg_data.coords['x1'] = np.arange(label_2.shape[0])
    seg_data.coords['y1'] = np.arange(label_2.shape[1])
    seg_data.coords['z1'] = np.arange(label_2.shape[2])
    seg_data.attrs['comment'] = 'gas phases: 1 in gas phase is the membrane, 2 the gas between CL and memrbane'
    seg_data.attrs['gas label gitsha'] = git_sha
    seg_data.attrs['cropping of seg data'] = (a1,b1,c1,d1,e1,f1)
    seg_data['gas_phases'] = (['x1','y1','z1','timestep'], label_2)
    seg_data['membrane_mask'] = (['x1','y1','z1','timestep'], mem_mask)
    seg_data['BPM_cav_volume'] = ('timestep', BPM_cav)
    seg_data['CL_cav_volume'] = ('timestep', CL_cav)
    
    seg_data.to_netcdf(segpath)
    
    
samples = list(segcropdict.keys())

for sample in samples:
    assign_labels(sample, segcropdict)
