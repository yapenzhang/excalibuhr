# File: src/excalibuhr/utils.py
__all__ = []

import os
import json
import pathlib
import numpy as np
import pandas as pd
from astroquery.eso import Eso
from astropy.io import fits
from astropy.stats import sigma_clip
from numpy.polynomial import polynomial as Poly
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')


def wfits(fname, im, hdr=None):
    """wfits - write image to file fname, 
    overwriting any old file"""
    primary_hdu = fits.PrimaryHDU(im, header=hdr)
    new_hdul = fits.HDUList([primary_hdu])
    new_hdul.writeto(fname, overwrite=True, output_verify='ignore') 

def order_trace(det, badpix, clip=10):
    im = np.where(badpix, signal.medfilt2d(det, 7), det)
    # plt.imshow(badpix)
    # plt.show()
    plt.imshow(im, vmin=1000, vmax=2e4)
    plt.show()
    im_grad = im[1:,:]-im[:-1,:]
    im_grad2 = im_grad[1:,:]-im_grad[:-1,:]
    im_grad3 = im_grad2[1:,:]-im_grad2[:-1,:]
    im_grad_clean = sigma_clip(im_grad, clip, maxiters=1, axis=0)
    for i in range(100):
        plt.plot(im_grad[:,i])
        plt.plot(im_grad_clean[:,i])
        # plt.axhline(np.nanmedian(im_grad[:,i]))
        plt.show()





def write2table(wave, flux, flux_err, snr, hdr, outfile):
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu])
    col1 = fits.Column(name='WAVE', format='D', array=wave)
    col2 = fits.Column(name='FLUX', format='D', array=flux)
    col3 = fits.Column(name='FLUX_ERR', format='D', array=flux_err)
    col4 = fits.Column(name='SNR', format='D', array=snr)
    cols = fits.ColDefs([col1, col2, col3, col4])
    hdul.append(fits.BinTableHDU.from_columns(cols))
    hdul.writeto(outfile, overwrite=True, output_verify='ignore')

def write4waveinclude(w0, w1, outfile):
    col1 = fits.Column(name='LOWER_LIMIT', format='D', array=w0)
    col2 = fits.Column(name='UPPER_LIMIT', format='D', array=w1)
    cols = fits.ColDefs([col1, col2])
    t = fits.BinTableHDU.from_columns(cols)
    t.writeto(outfile, overwrite=True, output_verify='ignore')

def PolyfitClip(xx, yy, dg, ww=[1.], clip=4., max_iter=10, \
                plotting=False, reject=False):
    """
    Perform weighted least-square polynomial fit,
    iterratively cliping pixels above a certain sigma threshold
    ----------
    Parameters
    ----------
    dg : degree of polynomial 
    ww : if provided, it includes the weights of each pixel.
    clip : sigma clip threshold
    max_iter : max number of iteration in sigma clip
    plotting : if True, plot fitting and thresholds
    reject : if True, also return the xx array after sigma clip
    ----------
    Returns
    ----------
    Polynomial fit params
    """
    if plotting:
        import matplotlib.pyplot as plt
    mask = (np.isnan(xx)) | (np.isnan(yy)) | (np.isinf(xx)) | (np.isinf(yy))
    xx = xx[~mask]
    yy = yy[~mask]
    if len(ww)<2:
        ww = np.ones_like(xx)
    if np.sum(mask) > 0.9*len(mask):
        return np.zeros(dg)
    ite=0
    while ite < max_iter:
        poly = np.polynomial.polynomial.polyfit(xx, yy, dg, w=ww)
        y_model = np.polynomial.polynomial.polyval(xx, poly)
        res = yy - y_model
        threshold = np.std(res)*clip
        if plotting and ite>0:
            plt.plot(yy)
            plt.plot(y_model)
            plt.show()
            plt.plot(res)
            plt.axhline(threshold)
            plt.axhline(-threshold)
            plt.show()
        if np.sum(np.abs(res) >= threshold) > 0.1:
            xx, yy, ww = xx[np.abs(res) < threshold], \
				yy[np.abs(res) < threshold], \
				ww[np.abs(res) < threshold]
        else:
            break
        ite+=1
    if reject: #return the xx without outliers
        return poly, xx
    else:
        return poly
