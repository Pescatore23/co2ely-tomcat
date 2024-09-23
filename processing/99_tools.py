# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:29:17 2022

@author: fische_r
"""

import xarray as xr
import numpy as np

def convert_4D_nc_to_3plus1D(path, destination, datakey='image_data'):
    '''
    

    Parameters
    ----------
    path : str
        path to nc file containing the 4D image data.
    destination : str
        path to new nc file containing the image data as 3+1D data, i.e. series of 3D.
    key : str
        name of the data array to be converted

    Returns
    -------
    None.

    '''
    data4D = xr.load_dataset(path)
    x = data4D['x'].data
    y = data4D['y'].data
    z = data4D['z'].data
    
    data3D = xr.Dataset({'image_data_00': (['x','y','z'], data4D[datakey][...,0].data),
                        },
                      coords = {
                          'x': x,
                          'y': y,
                          'z': z
                          },
                      )
    for i in range(data4D[datakey].shape[-1]):
        key = 'image_data_'+ f'{i:02d}'
        data3D[key] = (['x','y','z'], data4D[datakey][...,i].data)
        
    data3D.attrs = data4D.attrs
    data3D.to_netcdf(destination)
    data4D.close()
    return data3D

# TODO: method to convert 3+1D to 4D

def convert_3p1D_nc_to_4D(path, destination):
    # only works for the COELY Tomcat data
    
    data = xr.open_dataset(path)
    images = [im for im in data.keys() if im[3:7] == 'imag']
    images.sort()
    
    x = data['x'].data
    y = data['y'].data
    z = data['z'].data
    t_utc = data['t_utc'].data
    x_m = data['x_m'].data
    y_m = data['y_m'].data
    z_m = data['z_m'].data  
    time = data['time'].data
    ts = np.arange(len(time))
    
    shp = data[images[4]].shape
    # shp = data[images[4]][:, c:d, e:f].shape
    shp = shp + (len(images),)
    im = np.zeros(shp, dtype = np.uint16)
    
    for i in range(shp[-1]):
        if i%10 == 0:
            print(i)
        im[...,i] = data[images[i]].data
        
    data4d = xr.Dataset({'image_data': (['x','y','z', 'ts'], im),
                         'x_m': ('x', x_m),
                         'y_m': ('y', y_m),
                         'z_m': ('z', z_m),
                         'time': ('ts', time),
                         't_utc': ('ts', t_utc)},
                        coords = {'x': x,
                                  'y': y,
                                  'z': z,
                                  'ts': ts},
                         
                         )
    data4d.attrs = data.attrs
    data.close()
    data4d.to_netcdf(destination)


import cupy as cp
from time import sleep
def find_free_GPU_memory(gpu_id, limit=0.5, num_GPU=5):
    free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
    
    while free<limit:
        gpu_id = (gpu_id+1)%num_GPU #for all available GPUs
        sleep(2)
        free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
        
    return gpu_id
    
