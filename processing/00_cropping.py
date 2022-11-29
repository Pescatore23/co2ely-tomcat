# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:58:37 2022

@author: fische_r
"""
import xarray as xr
import robpylib
import h5py
import os
import numpy as np
import json
import subprocess

# TODO maybe include fetching git version in robpylib
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)

rawDataFolder = '/das/home/fische_r/SLCO2ELY/Data10/disk1'
baseFolder = '/das/home/fische_r/DASCOELY/Data10/disk1'
proc_stage = '00_raw'

#TODO: create an extra datafile, for now one entry for the test samples
crop_dict = {'T_3III_scan_05': [801, 801+867, 51, 51+849, 0, -1]} 

def get_scan_timestamps(sample, rawDataFolder, n0=1000):
    print(sample)
    sample_id = sample
    # metapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'_config.json'])) #old filename for interlaces
    metapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'.json'])) #"new" naming for COELY
    datapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'.h5']))
    
    if  os.path.exists(metapath):
        metadata = json.load(open(metapath,'r'))
        # n = metadata['nimages']
        n = metadata['scientificMetadata']['scanParameters']['Number of projections'] #new json convention
        
    else:
        print('config file not found, hard coded n='+str(n0))
        n = n0

    if not os.path.exists(datapath):
        print('h5 file not found')
        return None
    else:
        datafile = h5py.File(datapath, 'r')
        data = datafile['measurement']['instrument']['acquisition']['data']
        data = np.array(data)
        time_stamps = np.zeros(data.size)
        datafile.close()
        
        for i in range(data.size):
            time_stamps[i] = data[i][-2]
            t0 = data[i][-3]
            
        return time_stamps[::n]/1e7, t0


def crop_stack(Stack, a,b,c,d,e,f):
    Stack = Stack[a:b,c:d,e:f]
    return Stack

def load_and_crop_time_step(sampleFolder, stepFolder, croplist):
    # loads reconstruction stack entirely in RAM (roughly 16GB), dask could work around, but not worth the effort
    Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(sampleFolder,stepFolder))
    Stack = crop_stack(Stack, *croplist)
    return Stack, names
    


def create_h5(sample, baseFolder, rawDataFolder, cropdict):
    
    # TODO: put pre-operando scans in add. xarray Dataarrays within Dataset  
    # TODO: add EC-lab data and gas cues to unified datafile, maybe better smaller extra file depending on size of image data
    
    sample_id = sample #TODO same sample_id for all scan_0X
    croplist = cropdict[sample_id]['crops']
    locref = cropdict[sample_id]['reference location']
    locref = [0,0,0]
    
    time, t0 = get_scan_timestamps(sample, rawDataFolder)
    timestamps = time+t0
    
    newdim = list(croplist)+[len(timestamps)]
    newStack = np.zeros(newdim, dtype=np.uint16)
    
    sampleFolder = os.path.join(baseFolder, sample, proc_stage)
    steps = os.listdir(sampleFolder)
    stepnames = []
    
    for (stepFolder,i) in zip(steps, range(newdim[-1])):
        Stack, names = load_and_crop_time_step(sampleFolder, stepFolder, croplist)
        newStack[...,i] = Stack
        stepnames.append(names)
    # TODO: convert stepnemase to 2D array for xarray dataset
    
    px = 2.75e-6 #m
    # vx = px**3
    #TODO: common reference, i.e. edge of first channel, also rotate sample images
    # reference maybe after registration if applicable
    x = np.arange(newdim[0]) - locref[0]
    y = np.arange(newdim[1]) - locref[1]
    z = np.arange(newdim[2]) - locref[2]
    
    x_metric = x*px
    y_metric = y*px
    z_metric = z*px
    
    data = xr.Dataset({'image_data': (['x','y','z','t_utc'], newStack),
                       'time': ('t_utc', time),
                       'x_m': ('x', x_metric),
                       'y_m': ('y', y_metric),
                       'z_m': ('x', z_metric),
                       'filenames': (['z','t_utc'], stepnames)
                       },
                      coords = {
                          't_utc': timestamps,
                          'x': x,
                          'y': y,
                          'z': z
                          },
                      attrs = {'name': sample_id,
                               'voxel size': '2.75 um',
                               'voxel': px,
                               'crop coordinates': croplist,
                               'reference coordinates': locref, #TBD
                               'git_sha': git_sha,
                               'githash': githash
                               }
                      )
    return data
    
    # TODO rest
        
    
    

    
    
    
    
    
    
    