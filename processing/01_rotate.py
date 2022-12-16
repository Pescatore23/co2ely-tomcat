# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:46:19 2022

@author: fische_r
"""

import os
import scipy as sp
from scipy import ndimage
import numpy as np
import xarray as xr
import json
from joblib import Parallel, delayed

import socket
host = socket.gethostname()
import subprocess

px = 2.75E-6
njobs = 16

# paths, add locations if another machine, e.g. ra is used
if host == 'mpc2959.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    toppath = '/mpc/homes/fische_r/NAS/DASCOELY'
    temppath = '/mnt/SSD/fische_r/tmp/joblib_tmp'
        
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)
    
processing_path = os.path.join(toppath, 'processing')

path_00_3p1D = os.path.join(processing_path, '00_cropped_vis_in_FIJI')
path_01_rotated = os.path.join(processing_path, '00_rotated_3p1D')

if not os.path.exists(path_01_rotated):
    os.mkdir(path_01_rotated)

def rotate_time_scan(key, data, rot_params):
    rotangle = rot_params['angle']
    a,b,c,d,e,f = rot_params['crop_after_rot']
    im = data[key].data
    im = sp.ndimage.rotate(im, rotangle)
    im = im[a:b,c:d,e:f]
    return im
    

def rotate_scan(path, njobs=njobs):
    data = xr.open_dataset(path)
    attrs = data.attrs
    scan_nr = attrs['scan_#']
    sample = attrs['name']
    t_utc = data['t_utc'].data
    time = data['time'].data
    filenames = data['filenames'].data
    
    image_keys = list(data.keys())
    for key in image_keys():
        if not key[:4] == 'imag':
            image_keys.remove(key)
    
    new_keys = []
    for key in image_keys:
        new_keys.append(scan_nr+'_'+key)
        
    rot_info_path = os.path.join(gitpath,'data_analysis', 'image_rotation_parameter', ''.join(['rot_params_',sample,'.json']))
    rot_params = json.load(open(rot_info_path, 'r'))
    filenames = filenames[:,rot_params['crop_after_rot'][-2]:rot_params['crop_after_rot'][-1]]
    
    results = Parallel(n_jobs=njobs, temp_folder=temppath)(delayed(rotate_time_scan(key,data, rot_params)) for key in image_keys)
    data.close()
    
    return {'keys': new_keys, 'images': results, 't_utc': t_utc, 'time':time}, filenames, attrs, scan_nr, rot_params



def rotate_sample(sample):
    sample_files = []
    check = sample
    if sample == '3II':
        check = '3b'
    for file in os.listdir(path_00_3p1D):
            if file.split('_')[2] == sample or file.split('_')[2] == check:
                sample_files.append(os.path.join(path_00_3p1D, file))
    sample_files.sort()
    
    rotated_dict = {}
    for file in sample_files:
        rotated_dict[file] = rotate_scan(file)
        
    # TODO: concatenate t_utc, time, filenames
    t_utc = []
    time = []
    filenames = []
    for file in sample_files:
        t_utc_i = rotated_dict[file][0]['t_utc]']
        time_i = rotated_dict[file][0]['time']
        t_utc = t_utc + [t_utc_i]
        time = time_i + [time]
        filenames.append(rotated_dict[file][1])
        
    t_utc = np.concatenate(t_utc)
    time = np.concatenate(time)
    filenames = np.stack(filenames, axis= 0)
        
    # set up the dataset with one 3D scan
    key0 = rotated_dict[sample_files[0]][0]['keys'][0]
    im0 = rotated_dict[sample_files[0]][0]['images'][0]
    rot_params = rotated_dict[sample_files[0]][4]
    
    x = np.arange(im0.shape[0])
    y = np.arange(im0.shape[1])
    z = np.arange(im0.shape[2])
    
    x_metric = x*px
    y_metric = y*px
    z_metric = z*px
    
    data = xr.Dataset({key0: (['x','y','z'], im0),
                        'time': ('t_utc', time),
                        'x_m': ('x', x_metric),
                        'y_m': ('y', y_metric),
                        'z_m': ('z', z_metric),
                        'filenames': (['t_utc','z'], filenames)
                        },
                      coords = {
                          't_utc': time,
                          'x': x,
                          'y': y,
                          'z': z
                          }, 
                      attrs = {'name': sample,
                                'voxel size': '2.75 um',
                                'voxel': px,
                                'post rotation cropping coordinates [a:b,c:d,e:f]': rot_params['crop_after_rot'],
                                'rotation angle': rot_params['angle'],
                                'git_sha_rotation': git_sha,
                                'githash_rotation': githash
                                }
                              )
    
    # populate data with the remaining data
    for file in sample_files:
        scan_nr = rotated_dict[file][3]
        data.attrs[scan_nr+'_attrs'] = rotated_dict[file][2]
        result_dict = rotated_dict[file][0]
        for (key, im) in zip(result_dict['keys'], result_dict['images']):
            if not key in data.keys():
                data[key] = (['x','y','z'], im)
                
    return data
    
    
    