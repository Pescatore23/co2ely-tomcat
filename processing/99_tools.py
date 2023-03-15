# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:29:17 2022

@author: fische_r
"""

import xarray as xr

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
    data4D = xr.open_dataset(path)[datakey]
    x = data4D['x'].data
    y = data4D['y'].data
    z = data4D['z'].data
    
    data3D = xr.Dataset({'image_data_00': (['x','y','z'], data4D[...,0].data),
                        },
                      coords = {
                          'x': x,
                          'y': y,
                          'z': z
                          },
                      )
    for i in range(data4D.shape[-1]):
        key = 'image_data_'+ f'{i:02d}'
        data3D[key] = (['x','y','z'], data4D[...,i].data)
        
    data3D.attrs = data4D.attrs
    data3D.to_netcdf(destination)
    data4D.close()
    return data3D

# TODO: method to convert 3+1D to 4D
