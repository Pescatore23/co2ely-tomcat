# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:24:27 2023

@author: fische_r
"""

#modules
import xarray as xr
import json
import os
import socket
host = socket.gethostname()
import SimpleITK as sitk
import numpy as np
# from joblib import Parallel, delayed  ## simpleITK appearantly does not work with joblib embarassing parallelisation
import subprocess

samples = ['1', '3II', '3III', '4', '4II', '5II', '5', '6', '7x', '8x']
samples = ['4']

# paths
if host == 'mpc2959.psi.ch':
    gitpath = '/mpc/homes/fische_r/lib/co2ely-tomcat'
    toppathHDD = '/mpc/homes/fische_r/NAS/DASCOELY/processing'
    toppath = '/mnt/SSD/fische_r/COELY'
    temppath = '/mnt/SSD/fische_r/tmp/joblib_tmp'
    
# path_01_4D = os.path.join(toppath, '01_rotated_3p1D')
path_01_4D = os.path.join(toppathHDD, '01_abs_rotated_3p1D')
registration_mask_path = os.path.join(gitpath, 'processing', 'masking_for_registration.json')  
outpath_02  = os.path.join(toppath, '02_abs_registered_3p1D')
if not os.path.exists(outpath_02):
	os.mkdir(outpath_02)
      
# git hash
scriptpath = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(scriptpath)
git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
os.chdir(cwd)    

mask_dimension = json.load(open(registration_mask_path, 'r'))

##### Parameters for the registration   ################
parameterMap = sitk.GetDefaultParameterMap("rigid") #get default settins for rigid registration

# modify some parameters
parameterMap["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
parameterMap["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
parameterMap["NumberOfResolutions"] = ["2"]
parameterMap["MaximumNumberOfIterations"] = ["1500"]
parameterMap["AutomaticTransformInitialization"] = ["true"]
parameterMap["AutomaticScalesEstimation"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["Interpolator"] = ["BSplineInterpolator"]
parameterMap["NumberOfSamplesForExactGradient"] = ["10000"]
parameterMap["NumberOfSpatialSamples"] = ["10000"]

# copied from Salvo's and Mayank's code
# TODO: adapat for own purpose, i.e. masking, parallelization, affine registration, transformation matrix
def register_images_general(image_name, im_fixed, im_tomove, parameter_map, transpath, verbose=False, im_mask=None):

    r"""
    General registration for 2D or 3D images
    This function allows control over the parameter map
    
         
    Parameters
    ----------     
    im_fixed : numpy.ndarray
        Fixed image
    im_tomove: numpy.ndarray
        Image to register (to im_fixed)
    parameter_map: sitk.ParameterMap
        Parameters map object as obtained from sitk.GetParameterMap
    verbose : bool, optional (default=False)
        Enable/Disable log to console 
    im_mask : numpy.ndarray (default=None)
        Mask for sampling points (required). 
        im_mask has same dimensions as im_fixed 
        The mask ndarray has 1 and 0. 1 values denote the area where
        points for the registration are sampled.    
    Returns
    -------
    im_aligned : numpy.ndarray
        Aligned image having same dimensions of im_tomove
                 
    Notes
    -----
    To have further info on parameter maps, go to:
    https://simpleelastix.readthedocs.io/ParameterMaps.html
    For more details on the different parameters, download:
    https://elastix.lumc.nl/download/elastix_manual_v4.8.pdf
    """
    
    itk_image_fixed = sitk.GetImageFromArray(im_fixed)
    itk_image_tomove = sitk.GetImageFromArray(im_tomove)
           
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(itk_image_fixed)
    elastixImageFilter.SetMovingImage(itk_image_tomove)
    elastixImageFilter.SetParameterMap(parameter_map)
    
    if im_mask is not None:
        itk_image_mask = sitk.GetImageFromArray(im_mask)
        # Casting type just in case
        itk_image_mask = sitk.Cast(itk_image_mask, sitk.sitkUInt8)
        elastixImageFilter.SetFixedMask(itk_image_mask)
    
    # Disable/enable logging 
    elastixImageFilter.SetLogToConsole(verbose)
    elastixImageFilter.Execute()
    
    # save transformation vector
    vectorpath = os.path.join(transpath, image_name+'.txt')
    transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    sitk.WriteParameterFile(transformParameterMapVector[0], vectorpath)
    

    im_aligned = elastixImageFilter.GetResultImage()
    # Transfor again to numpy
    im_aligned = sitk.GetArrayFromImage(im_aligned)
    return np.uint16(im_aligned)

def transform_images(image_name, im_tomove, transpath, verbose=False):
    
    itk_image_tomove = sitk.GetImageFromArray(im_tomove)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetMovingImage(itk_image_tomove)
 
    # load and apply transformation vector
    vectorpath = os.path.join(transpath, image_name+'.txt')
    transformParameterMapVector = sitk.ReadParameterFile(vectorpath)
    im_aligned = sitk.Transformix(itk_image_tomove, transformParameterMapVector)
    
    # Transfor again to numpy
    im_aligned = sitk.GetArrayFromImage(im_aligned)   
    
    return np.uint16(im_aligned)


def register_sample(sample, apply_transform=True):
    
    # open data
    file = '01_'+sample+'_rotated_3p1D.nc'
    imagepath = os.path.join(path_01_4D, file)
    data = xr.load_dataset(imagepath)
    # px = data['voxel']
    transpath = os.path.join(scriptpath, 'transformation_vectors', sample)
    
    if not os.path.exists(transpath):
        os.mkdir(transpath)
    
    images = [im for im in data.keys() if im[3:7] == 'imag']
    images.sort()
    
    reference_image_name = images[5]
    
    im_fixed = data[reference_image_name].data
    
    mask = np.zeros(im_fixed.shape, dtype=np.uint8)
    a,b,c,d,e,f = mask_dimension[sample]
    mask[a:b,c:d,e:f] = 1 #
    
    if not apply_transform:
        print('calculating tranformation')
        results = [register_images_general(image_name, im_fixed, data[image_name].data, parameterMap, transpath, im_mask=mask) for image_name in images]
    else:
        print('applying transformation')
        results = [transform_images(image_name, data[image_name].data, parameterMap, transpath, im_mask=mask) for image_name in images]
    
    
    # set up the dataset with one 3D scan
    im0 = results[0]   
    data_reg = xr.Dataset({images[0]: (['x','y','z'], im0),
                      'time': ('t_utc', data['time'].data),
                      'x_m': ('x', data['x_m'].data),
                      'y_m': ('y', data['y_m'].data),
                      'z_m': ('z', data['z_m'].data),
                      'filenames': (['t_utc','z'], data['filenames'].data)},
                          coords = {
                              't_utc': data['t_utc'].data,
                              'x': data['x'].data,
                              'y': data['y'].data,
                              'z': data['z'].data,
                              }
        )
    data_reg.attrs = data.attrs.copy()
    data_reg.attrs['git_sha_registration'] = git_sha
    data_reg.attrs['githash_registration'] = githash
    
    # add remaining 3D scans
    for i in range(len(images)):
        data_reg[images[i]] = (['x','y','z'], results[i])
        
    return data_reg



for sample in samples:
    print(sample)
    print('rigid registration scans')
    data_reg = register_sample(sample)
    datafile = ''.join(['02_',sample,'_registered_3p1D_repeat.nc'])
    datapath = os.path.join(outpath_02, datafile)
    print('save file to disk')
    data_reg.to_netcdf(datapath)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
