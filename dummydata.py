#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:20:13 2020

@author: vand
"""

import numpy as np
import scipy.ndimage
import scipy.io
import scipy.ndimage.filters

def binary_splatty(dim, sigma=7, threshold=0, boundary=0):
    """ Splatty data is a test volume suitable for porosity analysis.
        This is an early version of random_organic_structure.
    Arguments:
        dim: tuple giving the size of the volume
        sigma: smoothing scale, higher value - smoother objects
        threshold: higher value - less material (smaller objects)
        boundary: strength of imposing object boundary pulled inwards
    Returns:
        a binary test volume
    Author: vand@dtu.dk, 2019
    """
    r2 = np.fromfunction(lambda x, y, z: (x/(dim[0]-1)-0.5)**2 + 
                (y/(dim[1]-1)-0.5)**2 + (z/(dim[2]-1)-0.5)**2, dim, dtype=int)
    B = np.random.standard_normal(dim)
    B[r2>1] -= boundary;
    B = scipy.ndimage.gaussian_filter(B, sigma, mode='constant',
                                      cval=-boundary)
    B = B>threshold
    B[[0,-1],:,:] = False
    B[:,[0,-1],:] = False
    B[:,:,[0,-1]] = False
    return B


def random_organic_structure(dim, gaussian_sigma=7, median_kernel= 9,
                             threshold=0.005, round_adjust=0.01,
                             box_adjust = 0.01, random_seed=0):
    """ Random organic structure is a test volume suitable for porosity analysis.
    Arguments:
        dim: tuple giving the size of the volume.
        gaussian_sigma: scalar, standard deviation of the Gaussian smoothing
            kernel, higher value - smoother objects.
        median_kernel: size of the median smooting kernel, higher value -
            smoother objects.
        threshold: threshold value, set clost to 0, higher value - less
            material (smaller objects).
        round_adjust: adjustment encouraging roundish shape.
        box_adjust: adjustment encouraging empty voxels at volume boundary.
        random_seed: int, random number generator seed
    Returns:
        a binary test volume
    Author: vand@dtu.dk, 2020
    """

    # Start with a random volume.
    np.random.seed(random_seed)
    V = np.random.standard_normal(dim)

    # adjusting values outside the elipsoid centered in a volume, to encourage
    # a roundish shape
    B = np.fromfunction(lambda x, y, z: ((2*x/(dim[0]-1)-1)**2 +
            (2*y/(dim[1]-1)-1)**2 + (2*z/(dim[2]-1)-1)**2) > 1, dim, dtype=int)
    V[B] -= round_adjust;

    # Smooth.
    V = scipy.ndimage.gaussian_filter(V, sigma=gaussian_sigma, mode='constant',
                                      cval=-box_adjust)
    V = V>threshold
    V = scipy.ndimage.median_filter(V, size=median_kernel, mode='constant',
                                    cval=False)

    # Impose a one-voxel empty boundary.
    V[[0,-1],:,:] = False
    V[:,[0,-1],:] = False
    V[:,:,[0,-1]] = False
    return V


def binary_to_intensities(B, intensities = [255/4, 3*255/4],
                          sigma = 2, noise = 20):
    '''
    Transforms binary image to intensity-like image with two materials given
    by intensities, smoothed by sigma, and corrupted by noise.
    Author: vand@dtu.dk, 2021
    '''

    intensities = np.asarray(intensities, dtype=np.uint8)
    V = intensities[B.astype(np.int)]
    V = scipy.ndimage.gaussian_filter(V, sigma, mode='wrap')
    V = V + noise*np.random.standard_normal(V.shape)
    V = np.minimum(V,255)
    V = np.maximum(V,0)
    V = V.astype(np.uint8)

    return V
