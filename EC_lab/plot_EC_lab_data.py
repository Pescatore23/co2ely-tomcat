# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:45:24 2022

@author: fische_r
"""

import json
import socket
import os
import matplotlib.pyplot as plt
from matplotlib import cm

# import the function to read mpr data, might fail if cwd wrong !
from read_EC_lab_data import mpr_to_xr

host = socket.gethostname()

metapath = None
if host == 'PC14918':
    metapath = r"C:\git_repos\01_Python\co2ely-tomcat\meta_and_linking_data\data_overview.json"
    EC_path = r"U:\07_X_ray\005_TOMCAT_2022\02_EC_lab_files"
    
    
overview = json.load(open(metapath, 'r'))


def load_sample_data(sample, overview=overview, EC_path=EC_path):
    
    EC_files = overview['samples'][sample]['ec_lab_files']
    EC_data_files = [file for file in EC_files if file[-3:]=='mpr'] #select only the data files
    
    data = []
    for file in EC_data_files:
        d1 = mpr_to_xr(os.path.join(EC_path, file))
        if not d1 is None:
            data.append(d1)    
        
    return data

def plot_vs_time(data, key):
    for d in data:
        d[key].plot()


def plot_HFR_time_color(d, xl=(0,200), yl=(-50,50), cmap=cm.viridis, tl=None, unit='min'):
    '''
    plots the HFR for one dataset with color coding for time
    '''
    if unit == 'min':
        tf = 60
    if unit == 's':
        tf = 1
    if tl is None:
        tl = (0,d['time'].max()/tf)
    plt.scatter(d['Re(Z)'], d['-Im(Z)'], c=(d['time'])/tf, cmap = cmap, marker='.', vmin=tl[0], vmax=tl[1])
    plt.colorbar(label='time ['+unit+']')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlim(xl[0],xl[1])
    plt.ylim(yl[0],yl[1])
    plt.xlabel('Re(Z) [$\Omega$]')
    plt.ylabel('-Im(Z) [$\Omega$]')

# TODO: HFR time color for multiple datafiles

# plt.axvline and plt.axhline draw lines across plot without need for limits
    
