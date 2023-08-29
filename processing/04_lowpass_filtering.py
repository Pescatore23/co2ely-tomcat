# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:08:07 2023

@author: fische_r
"""

import xarray as xr
import os
import socket
host = socket.gethostname()
import subprocess
import numpy as np
import cupy as cp
import cucim.skimage.filters
from joblib import delayed, Parallel


samples = ['3II', '3III', '4', '4II', '5II', '5', '6', '7x', '8x'] #'1'
# paths
if host == 'mpc2959.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    toppathHDD = '/mpc/homes/fische_r/NAS/DASCOELY/processing'
    toppath = '/mnt/SSD/fische_r/COELY'
    temppath = '/mnt/SSD/fische_r/tmp/joblib_tmp'
    
#path_01_4D = os.path.join(toppath, '01_abs_rotated_3p1D')
path_02_4D = os.path.join(toppath, '02_registered_3p1D')

outpath_04 = os.path.join(toppathHDD, '04_lowpass_filtered_4D')

if not os.path.exists(outpath_04):
	os.mkdir(outpath_04)
    
# git hash
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)    


def filter_subimage(im, gpu_id):
    with cp.cuda.Device(gpu_id):
        imgpu = cp.array(im)
        imgpu = cucim.skimage.filters.butterworth(imgpu, high_pass=False, cutoff_frequency_ratio=0.4)
        im = cp.asnumpy(imgpu)
        del imgpu
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    
    return im

def filter_image(im):
    d = int(im.shape[2]/3)   
    ims = [im[:,:,:d], im[:,:,d:2*d], im[:,:,2*d:]] 
    imfilts = Parallel(n_jobs=3, temp_folder=temppath)(delayed(filter_subimage)(ims[i], i+1) for i in range(3))
    im = np.concatenate(imfilts, axis=2)
    return im
    
def filter_sample(sample):
   # open data
    file = '02_'+sample+'_registered_3p1D.nc'
    imagepath = os.path.join(path_02_4D, file)
    data = xr.load_dataset(imagepath)
   
    images = [im for im in data.keys() if im[3:7] == 'imag']
    images.sort()
    
    shpt = len(images)
    shp = data[images[0]].shape + (shpt,)
    
    im4D = np.zeros(shp, dtype=np.uint16)
    
    for i in range(shpt):
        if i%10==0: print(str(i),'/',str(shpt))
        im = data[images[i]]
        im = filter_image(im)
        im4D[...,i] = im
        
    data_filt = xr.Dataset({'image_data': (['x','y','z','t_utc'], im4D),
                          'time': ('t_utc', data['time'].data),
                          'x_m': ('x', data['x_m'].data),
                          'y_m': ('y', data['y_m'].data),
                          'z_m': ('z', data['z_m'].data),
                          'filenames': (['t_utc','z'], data['filenames'].data)},
                              coords = {
                                  't_utc': data['t_utc'].data,
                                  'x': data['x'].data,
                                  'y': data['y'].data,
                                  'z': data['z'].data,
                                  }
                              )
    data_filt.attrs = data.attrs.copy()
    data_filt.attrs['git_sha_filtering'] = git_sha
    data_filt.attrs['githash_filtering'] = githash
    
    return data_filt

for sample in samples:
    print(sample)
    print('lowpass filtering scans')
    data_filt = filter_sample(sample)
    datafile = ''.join(['04_',sample,'_lowpass_filteres_4D.nc'])
    datapath = os.path.join(outpath_04, datafile)
    print('save file to disk')
    data_filt.to_netcdf(datapath)
    
    
