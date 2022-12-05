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

# import argparse

# TODO maybe include fetching git version in robpylib
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)

rawDataFolder = '/das/home/fische_r/SLSCOELY'
baseFolder = '/das/home/fische_r/DASCOELY/Data10/disk1'
proc_stage = '00_raw'
destination = '/das/home/fische_r/DASCOELY/processing'

#TODO: create an extra datafile, for now one entry for the test samples
# cropdict = {'T_3III_scan_04': {'crops': [792, 792+834, 45, 45+856, 0, 2016]}} 

croppath = '/das/home/fische_r/DASCOELY/processing/cropping.nc'
cropdata = xr.load_dataset(croppath)['cropdict']


sample = 'T_3III_scan_04'


# TODO: find a way to use parser with job submission,e.g. check h5_reconstruct.py
# parser = argparse.ArgumentParser(description = 'Simulation parameters')

# parser.add_argument('-s', '--sample', type = str, default = None, help = 'The name of the sample, same as the corresponding TOMCAT folder')
# args = parser.parse_args()


# sample = args.sample
# testpath = os.path.join(baseFolder, ''.join([sample,'_test.nc']))
dest1 = os.path.join(destination, ''.join(['00_',sample,'_cropped_4D.nc']))
dest2 = os.path.join(destination, ''.join(['00_',sample,'_cropped_3Dp1D.nc']))


def get_scan_timestamps(sample, rawDataFolder, n0=1000, offset_proj=25):
    print(sample)
    sample_id = sample
    # metapath = os.path.join(rawDataFolder, sample_id, ''.join([sample_id,'_config.json'])) #old filename for interlaces
    metapath = os.path.join(rawDataFolder, sample_id, ''.join([sample_id,'.json'])) #"new" naming for COELY
    datapath = os.path.join(rawDataFolder, sample_id, ''.join([sample_id,'.h5']))
    
    if  os.path.exists(metapath):
        metadata = json.load(open(metapath,'r'))
        # n = metadata['nimages']
        n = metadata['scientificMetadata']['scanParameters']['Number of projections'] #new json convention
        n = n + offset_proj
        scans = metadata['scientificMetadata']['scanParameters']['Number of scans']
        
    else:
        print('config file not found, hard coded n='+str(n0))
        n = n0 + offset_proj
        scans = 0
    print('Warning: cut-off image offset hard-coded: '+str(offset_proj))

    if not os.path.exists(datapath):
        print('h5 file not found')
        return None
    else:
        datafile = h5py.File(datapath, 'r')
        data = datafile['measurement']['instrument']['acquisition']['data']
        
        if not scans*n == datafile['exchange']['data'].shape[0]:
            'Warning: multiple of scans and projection number do not match'
        
        data = np.array(data)
        time_stamps = np.zeros(data.size)
        datafile.close()
        
        for i in range(data.size):
            time_stamps[i] = data[i][-2]
            t0 = data[i][-3]
        time_stamps = time_stamps[::n]/1e7 #look at the time stamp of every nth projection
        
        
        return time_stamps, t0


def crop_stack(Stack, a,b,c,d,e,f):
    Stack = Stack[a:b,c:d,e:f]
    return Stack

def load_and_crop_time_step(sampleFolder, stepFolder, croplist):
    # loads reconstruction stack entirely in RAM (roughly 16GB), dask could work around, but not worth the effort
    Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(sampleFolder,stepFolder), track=False)
    Stack = crop_stack(Stack, *croplist)
    return Stack, names
    


def create_h5(sample, baseFolder, rawDataFolder, cropdata):
    
    # TODO: put pre-operando scans in add. xarray Dataarrays within Dataset, maybe not, easier to have a database that links the files
    # TODO: add EC-lab data and gas cues to unified datafile, maybe better smaller extra file depending on size of image data; same as above
    
    sample_id = sample.split('_')[1]
    if sample_id == '3b':
        sample_id = '3II'
    
    croplist = cropdata.sel(sample_id)
    a, b, c, d, e, f = croplist
    
    time, t0 = get_scan_timestamps(sample, rawDataFolder)
    timestamps = time+t0
 
    newdim = [b-a, d-c, f-e] +[len(timestamps)]
    newStack = np.zeros(newdim, dtype=np.uint16)
    
    sampleFolder = os.path.join(baseFolder, sample, proc_stage)
    steps = os.listdir(sampleFolder)
    steps.sort()
    stepnames = []
    

    i = 0
    total_i = len(steps)
    for stepFolder in steps:
        print('time step '+str(i+1)+'/'+str(total_i))
        Stack, names = load_and_crop_time_step(sampleFolder, stepFolder, croplist)
        newStack[...,i] = Stack
        stepnames.append(names)
        i = i+1
    stepnames = np.array(stepnames)
    
    px = 2.75e-6 #m
    # vx = px**3
    #TODO: common reference, i.e. edge of first channel, also rotate sample images
    # reference maybe after registration if applicable
    x = np.arange(newdim[0])# - locref[0]
    y = np.arange(newdim[1])# - locref[1]
    z = np.arange(newdim[2])# - locref[2]
    
    x_metric = x*px
    y_metric = y*px
    z_metric = z*px
    
    data = xr.Dataset({'image_data': (['x','y','z','t_utc'], newStack),
                        'time': ('t_utc', time),
                        'x_m': ('x', x_metric),
                        'y_m': ('y', y_metric),
                        'z_m': ('z', z_metric),
                        'filenames': (['t_utc','z'], stepnames)
                        },
                      coords = {
                          't_utc': timestamps,
                          'x': x,
                          'y': y,
                          'z': z
                          },
                      attrs = {'name': sample_id,
                               'foldername': sample,
                               'scan_#': sample.split('_')[-1],
                                'voxel size': '2.75 um',
                                'voxel': px,
                                'crop coordinates [a:b,c:d,e:f]': croplist,
                                'cropping:' 
                                # 'reference coordinates': locref, #TBD
                                'git_sha': git_sha,
                                'githash': githash
                                }
                      )

# create dataset with one time step as data reference
    data2 = xr.Dataset({'image_data_00': (['x','y','z'], newStack[...,0]),
    #                     'image_data_1': (['x','y','z'], newStack[...,1]),
                        'time': ('t_utc', time),
                        'x_m': ('x', x_metric),
                        'y_m': ('y', y_metric),
                        'z_m': ('z', z_metric),
                        'filenames': (['t_utc','z'], stepnames)
                        },
                      coords = {
                          't_utc': timestamps,
                          'x': x,
                          'y': y,
                          'z': z
                          },
                      )
    data2.attrs = data.attrs
    # add the remaining datarrays for each timestep
    for i in range(1,newStack.shape[-1]):
        key = 'image_data_'+ f'{i:02d}'
        data2[key] = (['x','y','z'], newStack[...,i])
    
    return data, data2

def run():
    data, data2 = create_h5(sample, baseFolder, rawDataFolder, cropdata)
    data.to_netcdf(dest1)
    data2.to_netcdf(dest2)
    
if not sample is None:  
    run()
else:
    'no sample name given'
        
    
    

    
    
    
    
    
    
    
