# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:19:13 2022

@author: fische_r
"""


import eclabfiles as ecf
import xarray as xr
import os
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

folder = r"U:\02_Electrolyser\006_testing\2022_10_05_variable_gas_flow"
folder = r"U:\02_Electrolyser\006_testing\2022_10_26_cell012"

def mpr_to_xr(path):
    _, meta = ecf.process(path)
    
    df = ecf.to_df(path)
    
    data = xr.Dataset(df)
    
    variables = list(data.keys())
    
    for var in variables:
        if var == 'uts':
            data[var].attrs['unit'] = 's'    
        else:
            if var in meta['units'].keys():
                data[var].attrs['unit'] = meta['units'][var]
    data['dim_0'] = data['time']
    if '<I>' in data.keys():
        data = data.rename({'dim_0': 't', '<I>': 'I', '<Ewe>': '$E_{we}$', 'uts': 'timestamp (utc)'})   
    else:
        data = data.rename({'dim_0': 't', 'Ewe': '$E_{we}$', 'uts': 'timestamp (utc)'})   
    data.attrs['path'] = path
    return data

# files = []
# big_data = []
# for file in os.listdir(folder):
#     if file[-3:] == 'mpr':
#         path = os.path.join(folder, file)
#         big_data.append(mpr_to_xr(path))
#         files.append(file)
        
        
        
# EIS = data['Re(Z)']>0
# peaks = sp.signal.find_peaks(EIS)
# fig, ax = plt.subplots(figsize = (16,9))
# ax.plot(data['t']/60, data['I'], 'k', linewidth=1)
# ax.set_xlabel('time [min]')
# ax.set_ylabel('I [mA]')

# ax2 = ax.twinx()
# ax2.plot(data['t']/60, EIS, 'k', alpha = 0.2)

# plt.savefig(figpath, dpi=600)



# i = 0
# for i in range(1,len(peaks[0])):
#     peak0 = peaks[0][i-1]-100
#     peak1 = peaks[0][i]-100
#     j = i-1
#     if j > 19:
#         j = i-21
#     if j > 19:
#         j = j-20
#     print(j)
#     plt.plot(data['Re(Z)'][peak0:peak1], data['-Im(Z)'][peak0:peak1], '.', color=cm.tab20(j))
    
    
# plt.xlim(0,700)
# plt.ylim(0,200)
# plt.xlabel('Re(Z) [$\Omega$]')
# plt.ylabel('-Im(Z) [$\Omega$]')
# plt.savefig(figpath2, dpi=600)


# fig, ax = plt.subplots(figsize = (12,5))
# ax2 = ax.twinx()
# ax3 = ax.twinx()

# for (data, file) in zip(big_data, files):
#     fig, ax = plt.subplots(figsize = (12,7))
#     ax2 = ax.twinx()
#     ax3 = ax.twinx()

#     EIS = data['Re(Z)']>0
#     peaks = sp.signal.find_peaks(EIS)
    
#     ax.plot(data['t']/60, data['I'], 'k', linewidth=1)
#     ax.set_xlabel('time [min]')
#     ax.set_ylabel('I [mA]')
    
#     ax2.plot(data['t']/60, data['$E_{we}$'], 'r', linewidth=1)
#     ax2.set_xlabel('time [min]')
#     ax2.set_ylabel('$E_{we}$ [V]')
        
#     ax3.plot(data['t']/60, EIS, 'k', alpha = 0.2)
    
#     ax.set_ylim(-10,0)
#     ax2.set_ylim(-4,-1)
#     ax3.set_yticks([])
#     plt.title(file)
    
#     figpath = os.path.join(r"U:\02_Electrolyser\006_testing\Cell012_plots",file[:-4]+'png')
#     plt.savefig(figpath, dpi=600)

# plt.savefig(figpath, dpi=600)
