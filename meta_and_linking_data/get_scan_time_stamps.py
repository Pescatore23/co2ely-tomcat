# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:09:34 2021

@author: firo
"""

import h5py
import os
import numpy as np
import json
# import pickle

# baseFolder = r"/Volumes/Volume/disk1_2"
# targetFolder = r"/Users/robfisch/NAS"

# samples = os.listdir(baseFolder)
# samples = os.listdir(targetFolder)


# TIME = {}
# 

def get_scan_timestamps(sample, baseFolder, n0=1000):
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
        
        for i in range(data.size):
            time_stamps[i] = data[i][-2]
            t0 = data[i][-3]
            
        return time_stamps[::n]/1e7, t0
