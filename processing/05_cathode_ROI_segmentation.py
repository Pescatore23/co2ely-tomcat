# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:38:33 2024

@author: fische_r
"""

import os
import xarray as xr
import socket
import subprocess
import numpy as np
import cupy as cp 
import cucim
from cucim.skimage.morphology import ball as GPUball
from scipy import ndimage
import cupyx.scipy.ndimage as GPUndimage
import cucim.skimage.morphology as cumorph
import skimage
from joblib import Parallel, delayed

#paths
host = socket.gethostname()
if host == '':
    gitpath = ''
    toppath = ''
    toppathSSD = ''
    temppath = ''
    temppath_2 = ''
    training_path = ''
    
elif host == '':
    gitpath = ''
    toppath = ''
    toppathSSD = ''
    temppath = ''
    temppath_2 = ''
    training_path = ''
else:
    print('host '+host+' currently not supported')


# fetch githash
cwd = os.getcwd()
os.chdir(gitpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)


crops = {   #part with GDL, necessary to reliably detect channels
'1': (180,-310),
'4': (280,-250),
'3II': (140,1620),
'3III': (40,1370),
'5': (160, 1580),
'6': (20, 1540),
'7x': (20,1260)
}

def get_GDL_mask(segim):
    fiber = segim == 0
    fibercp = cp.array(fiber)
    fibercp = GPUndimage.binary_opening(fibercp, structure=GPUball(1))
    dst = ndimage.distance_transform_cdt(~cp.asnumpy(fibercp))
    fibercp = cp.array(dst<3)
    labels = cucim.skimage.measure.label(fibercp)
    fibercp = labels == cp.argmax(cp.bincount(labels[fibercp]))
    del labels
    fibercp = GPUndimage.binary_erosion(fibercp, structure=GPUball(3))
    fibercp = GPUndimage.binary_dilation(fibercp, structure=GPUball(6))
    fibercp = GPUndimage.binary_fill_holes(fibercp)
    fibercp = GPUndimage.binary_erosion(fibercp, structure=GPUball(3))
    mask = cp.asnumpy(fibercp)
    del fibercp
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    return mask

def get_channel_masks(segim, mask, a, b):
    back = segim == 3
    FFcp = cp.array(back)
    FFcp = GPUndimage.binary_opening(FFcp, GPUball(4))
    
    FFcp = FFcp + cp.array(mask)
    FFcp[:5,:] = True
    FFcp[-5:,:] = True
    FFcp[:,:5] = True
    FFcp[:,-5:] = True
    FFcp = GPUndimage.binary_closing(FFcp, GPUball(4))
    FFcp[:5,:] = True
    FFcp[-5:,:] = True
    FFcp[:,:5] = True
    FFcp[:,-5:] = True
    cut = int(FFcp.shape[0]/2)
    FFcp[cut,:,:] = True
    FFlabels = cucim.skimage.measure.label(~FFcp[:,:,a:b], connectivity=2)
    FFcp[...] = False
    id, count = cp.unique(FFlabels, return_counts = True)
    id1,id2 = id[cp.argsort(count)][-3:-1]
    FFcp[:,:,a:b] = FFlabels==int(id1)

    channel1 = cp.asnumpy(FFcp)
    FFcp[...] = False
    FFcp[:,:,a:b] = FFlabels==int(id2)
    channel2 = cp.asnumpy(FFcp)

    del FFcp, FFlabels
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    # check which channel is on top and assign accordingly
    if np.argmax(channel1[:,:,650].sum(axis=1))>np.argmax(channel2[:,:,650].sum(axis=1)):
        return channel1, channel2
    else:
        return channel2, channel1

def get_channel_GDL_masks(channel1, channel2,a,b):
    GDLchannel1 = channel1.sum(axis=1)>100
    GDLchannel2 = channel2.sum(axis=1)>100
    GDLland = ~np.bitwise_or(GDLchannel1, GDLchannel2)
    GDLland[:,:a] = False
    GDLland[:,b:] = False
    labels = skimage.measure.label(GDLland)
    GDLrib = labels == np.argmax(np.bincount(labels[GDLland]))
    
    return GDLchannel1, GDLchannel2, GDLland, GDLrib

def get_timestep_mask(ts, seg_data, a,b):
    test = seg_data['segmented'].sel(timestep = ts)

    gpu_id = ts%5
    # while cp.cuda.Device(gpu_id).mem_info[0]<8E9:
    #     time.sleep(1)
    
    with cp.cuda.Device(gpu_id):
        mask = get_GDL_mask(test)
        channel1, channel2 = get_channel_masks(test, mask, a, b)
        GDLchannel1, GDLchannel2, GDLland, GDLrib = get_channel_GDL_masks(channel1, channel2,a,b)
        mask = mask.astype(np.uint8)

    segments = np.zeros(mask.shape, dtype = np.uint8)
    segments[channel1] = 1
    segments[channel2] = 2
    segments = segments + GDLchannel1[:,None,:].astype(np.uint8)*3*mask
    segments = segments + GDLchannel2[:,None,:].astype(np.uint8)*4*mask
    segments = segments + GDLland[:,None,:].astype(np.uint8)*6*mask
    segments = segments - GDLrib[:,None,:].astype(np.uint8)*1*mask
    

    return segments, mask
    
def all_segments(seg_data, a, b,):
    timesteps = seg_data['timestep'].data
    results = Parallel(n_jobs = 5, temp_folder=temppath)(delayed(get_timestep_mask)(ts, seg_data, a,b) for ts in timesteps)
    segments = [result[0] for result in results]
    mask = [result[0] for result in results]
    segments = np.stack(segments, axis=-1).astype(np.uint8)
    mask = np.stack(mask, axis=-1).astype(np.uint8)
    return segments, mask


#main
sample = '7x'
segpath = os.path.join(toppathSSD,'05_water_GDL_ML', sample, sample+'water_segmentation.nc')
seg_data = xr.open_dataset(segpath)
a, b = crops[sample]
segments, mask = all_segments(seg_data, a,b)

shp = segments.shape
ROIdata = xr.Dataset({'ROI': (['x','y','z','timestep'], segments),
                      'GDL': (['x','y','z','timestep'], mask),
                     't_utc': ('timestep',seg_data['t_utc'].data),
                     'time': ('timestep',seg_data['time'].data)},
                               coords = {'x': np.arange(shp[0]),
                               'y': np.arange(shp[1]),
                               'z': np.arange(shp[2]),
                               'timestep': np.arange(shp[3])
                               })
ROIdata.attrs = seg_data.attrs
ROIdata.attrs['ROI values'] = [1,2,3,4,5,6]
ROIdata.attrs['crops'] = [a,b]
ROIdata.attrs['ROI'] = ['channel1', 'channel2', 'GDLchannel1', 'GDLchannel2', 'GDLrib', 'GDL_other_land']

ROIpath = os.path.join(toppathSSD, '05_water_GDL_ML', sample, sample+'_ROI_segments.nc')
ROIdata.to_netcdf(ROIpath)