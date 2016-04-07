#!/usr/bin/python

import numpy as np

# histogram, re-compute cuts

def calcHistogram(imdata, bins_=256):
    '''Compute image histogram.'''
    im_values =  np.ravel(imdata)
    hh, bins_ = np.histogram( im_values, bins=bins_ )
    return hh, bins_

def calcDisplayRange(imdata, cutoff=.1, bins_=512):
    '''Compute display range, i.e., cuts.
    (ignore the 'cutoff'% lowest/highest value pixels)'''
    # compute image histogram
    hh, bins_ = calcHistogram(imdata, bins_)
    # check histogram format
    if len(bins_)==len(hh):
        bb = bins_
    else:
        bb = bins_[:-1]    # 'bins' == bin_edges
       # number of pixels
    Npx = np.sum(hh)
    # calc. lower limit : 
    i = 1
    while np.sum( hh[:i] ) < Npx*cutoff/100.:
        i += 1
    cut0 = round( bb[i] )
    # calc. higher limit
    j = 1
    while np.sum( hh[-j:] ) < Npx*cutoff/100.:
        j += 1
    cut1 = round( bb[-j] )
    return cut0,cut1
