from __future__ import division
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import rescale
import os

def crop(im): #crops 224 x 224 randomly from a bigger sized image
    sx, sy = im.shape[0:2]
    rx = np.random.randint(112,sx-112)
    ry = np.random.randint(112,sy-112)
    return im[rx-112:rx+112,ry-112:ry+112]

def preprocess(im, dim): #dim>224, it rescales such that the smaller axis is dim (ex.256), then it crops randomly. It does not sample (crop) multiple times for large aspect ratio.
    x = im.shape[0]
    y = im.shape[1]
    asp = max(y/x, x/y)
    s = dim/min(x,y)
    return crop(rescale(im,s))
    #an alternative to make a crop from bigger image is to pad the smaller image,
    #but crop is more convenient and I don't know what will happen if a plain padding dominates an image or happens more with certain category.

def onebyone(path): #resize/crops images in the directory to 224 x 224 and saves in the subfolder 'processed'.
    files = filter(None,map(lambda x: x if not os.path.isdir(path+x) else None, os.listdir(path)))
    if not os.path.isdir(path+'processed/'):
        os.mkdir(path+'processed/')
    i=1
    for f in files:
        print i,'/',len(files)
        if not os.path.isfile(path+'processed/'+f):
            data = preprocess(imread(path+f),256)
            imsave(path+'processed/'+f,data)
        i+=1
