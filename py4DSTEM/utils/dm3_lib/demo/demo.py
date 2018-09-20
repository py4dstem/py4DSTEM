#!/usr/bin/python

from __future__ import print_function, division

import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import dm3_lib as dm3

from utilities import calcHistogram, calcDisplayRange

# CONSTANTS

savedir = os.path.expanduser("~/Desktop")
debug = 0

# define command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("file", help="path to DM3 file to parse")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--dump", help="dump DM3 tags in text file",
                    action="store_true")
parser.add_argument("--convert", help="save image in various formats",
                    action="store_true")

# parse command line arguments
args = parser.parse_args()
if args.verbose:
    debug = 1
filepath = args.file

# get filename
filename = os.path.split(filepath)[1]
fileref = os.path.splitext(filename)[0]

# pyplot interactive mode
plt.ion()
plt.close('all')

# parse DM3 file
dm3f = dm3.DM3(filepath, debug=debug)

# get some useful tag data and print
print("file:", dm3f.filename)
print("file info.:")
print(dm3f.info)
print("scale: %.3g %s/px"%dm3f.pxsize)
cuts = dm3f.cuts
print("cuts:",cuts)

# dump image Tags in txt file
if args.dump:
    dm3f.dumpTags(savedir)

# get image data
aa = dm3f.imagedata

# display image
# - w/o cuts
if args.verbose:
    plt.matshow(aa, cmap=plt.cm.pink)
    plt.title("%s (w/o cuts)"%filename)
    plt.colorbar(shrink=.8)
# - w/ cuts (if cut values different)
if cuts[0] != cuts[1]:
    plt.matshow(aa, cmap=plt.cm.pink, vmin=cuts[0], vmax=cuts[1])
    plt.title("%s"%filename)
    plt.colorbar(shrink=.8)

# - display image histogram
if args.verbose:
    hh,bb = calcHistogram(aa)
    plt.figure('Image histogram')
    plt.plot(bb[:-1],hh,drawstyle='steps')
    plt.xlim(bb[0],bb[-1])
    plt.xlabel('Intensity')
    plt.ylabel('Number')

# convert image to various formats
if args.convert:
    # save image as TIFF
    tif_file = os.path.join(savedir,fileref+'.tif')
    im = Image.fromarray(aa)
    im.save(tif_file)
    # check TIFF dynamic range
    tim = Image.open(tif_file)
    if tim.mode == 'L':
        tif_range = "8-bit"
    else:
        tif_range = "32-bit"
    print("Image saved as %s TIFF."%tif_range)

    # save image as PNG and JPG files
    # - normalize image for conversion to 8-bit
    aa_norm = aa.copy()
    # -- apply cuts (optional)
    if cuts[0] != cuts[1]:
        aa_norm[ (aa <= min(cuts)) ] = float(min(cuts))
        aa_norm[ (aa >= max(cuts)) ] = float(max(cuts))
    # -- normalize
    aa_norm = (aa_norm - np.min(aa_norm)) / (np.max(aa_norm) - np.min(aa_norm))
    # -- scale to 0--255, convert to (8-bit) integer
    aa_norm = np.uint8(np.round( aa_norm * 255 ))

    if args.verbose:
        # - display normalized image
        plt.matshow(aa_norm, cmap=plt.cm.Greys_r)
        plt.title("%s [8-bit display]"%filename)
        plt.colorbar(shrink=.8)

    # - save as PNG and JPG
    im_dsp = Image.fromarray(aa_norm)
    im_dsp.save(os.path.join(savedir,fileref+'.png'))
    im_dsp.save(os.path.join(savedir,fileref+'.jpg'))

