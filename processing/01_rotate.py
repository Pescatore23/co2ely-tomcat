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
import argparse
import socket
host = socket.gethostname()
import subprocess

px = 2.75E-6
# njobs = 32
samples = ['1', '3II', '3III', '4', '4II', '5II', '5', '6', '7x', '8x']
# samples = ['4'] #samples to redo


parser = argparse.ArgumentParser()
parser.add_argument('-db', '--debug', type = str, default = 'no', help = 'debug or not, default Fno, option yes')
parser.add_argument('-n', '--njobs', type = str, default = 64, help = 'how many threads to use, default 64')
parser.add_argument('-s', '--sample', type = str, default = None, help = 'rotate just the specified sample, default None')

args = parser.parse_args()

# TODO: debug might not work this way, input may have to be converted from str
debug = False
if args.debug == 'yes':
    debug = True
njobs = args.njobs
if args.sample is not None:
    samples = [args.sample]
print(samples)
print(debug)

# paths, add locations if another machine, e.g. ra is used
if host == 'mpc2959.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    # toppath = '/mpc/homes/fische_r/NAS/DASCOELY'
    toppath = '/mnt/SSD/fische_r/COELY'
    temppath = '/mnt/SSD/fische_r/tmp/joblib_tmp'
    
elif host[:2] == 'ra':
    gitpath = '/das/home/fische_r/lib/coely'
    toppath = '/das/home/fische_r/DASCOELY/processing'
    temppath = None
else:
    print('host '+host+' not known')
        
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)
    
# processing_path = os.path.join(toppath, 'processing')
processing_path = toppath

path_00_3p1D = os.path.join(processing_path, '00_cropped_3p1D')
path_01_rotated = os.path.join(processing_path, '01_abs_rotated_3p1D')

if not os.path.exists(path_01_rotated):
    os.mkdir(path_01_rotated)

def rotate_time_scan(key, data, rot_params):
    rotangle1 = rot_params['angle1']
    rotangle2 = rot_params['angle2']
    a,b,c,d,e,f = rot_params['crop_after_rot']
    im = data[key].data
    im = sp.ndimage.rotate(im, rotangle1)
    im = sp.ndimage.rotate(im, rotangle2, axes = (2,1))
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
    
    data_keys = list(data.keys())
    image_keys = []
    for key in data_keys:
        if key[:4] == 'imag':
            image_keys.append(key)
    
    new_keys = []
    for key in image_keys:
        new_keys.append(scan_nr+'_'+key)
        
    rot_info_path = os.path.join(gitpath,'data_analysis', 'image_rotation_parameter', ''.join(['rot_params_',sample,'.json']))
    rot_params = json.load(open(rot_info_path, 'r'))
    filenames = filenames[:,rot_params['crop_after_rot'][-2]:rot_params['crop_after_rot'][-1]]
    
    # results = Parallel(n_jobs=njobs, temp_folder=temppath)(delayed(rotate_time_scan)(image_keys[i], data, rot_params) for i in range(len(image_keys)))
    results = Parallel(n_jobs=njobs, temp_folder=temppath)(delayed(rotate_time_scan)(key, data, rot_params) for key in image_keys)
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
        print(file)
        rotated_dict[file] = rotate_scan(file)
    
    print('calclulated everything, putting it in dataset ... ')
    # TODO: concatenate t_utc, time, filenames
    t_utc = []
    time = []
    filenames = []
    for file in sample_files:
        t_utc_i = rotated_dict[file][0]['t_utc']
        time_i = rotated_dict[file][0]['time']
        t_utc = t_utc + [t_utc_i]
        time = time + [time_i]
        filenames.append(rotated_dict[file][1])
        
    t_utc = np.concatenate(t_utc)
    time = np.concatenate(time)
    filenames = np.concatenate(filenames, axis= 0)
        
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
                          't_utc': t_utc,
                          'x': x,
                          'y': y,
                          'z': z
                          }, 
                      attrs = {'name': sample,
                                'voxel size': '2.75 um',
                                'voxel': px,
                                'post rotation cropping coordinates [a:b,c:d,e:f]': rot_params['crop_after_rot'],
                                'rotation angle 1': rot_params['angle1'],
                                'rotation angle 2': rot_params['angle2'],
                                'git_sha_rotation': git_sha,
                                'githash_rotation': githash,
                                'image_data_names': '<scan>_iamge_data_<time_step>, e.g. 02_image_data_00 is the first time step of scan 02'
                                }
                              )
    
    # populate data with the remaining data
    for file in sample_files:
        scan_nr = rotated_dict[file][3]
        data.attrs[scan_nr+'_crop_git_sha'] = rotated_dict[file][2]['cropping:git_sha']
        data.attrs[scan_nr+'_crop_githash'] = rotated_dict[file][2]['githash']
        result_dict = rotated_dict[file][0]
        for (key, im) in zip(result_dict['keys'], result_dict['images']):
            if not key in data.keys():
                data[key] = (['x','y','z'], im)
                
    return data
    
    
def run_and_save_data(samples):
    for sample in samples:
        print(sample)
        data = rotate_sample(sample)
        datafile = ''.join(['01_',sample,'_rotated_3p1D.nc'])
        datapath = os.path.join(path_01_rotated, datafile)
        data.to_netcdf(datapath)
        
if not debug:
    run_and_save_data(samples)
