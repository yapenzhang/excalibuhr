# File: src/excalibuhr/utils.py
__all__ = []

import os
import json
import pathlib
import numpy as np
import pandas as pd
from astroquery.eso import Eso
from astropy.io import fits
from astropy import stats
from numpy.polynomial import polynomial as Poly
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
import warnings


def wfits(fname, im, hdr=None):
    """wfits - write image to file fname, 
    overwriting any old file"""
    primary_hdu = fits.PrimaryHDU(im, header=hdr)
    new_hdul = fits.HDUList([primary_hdu])
    new_hdul.writeto(fname, overwrite=True, output_verify='ignore') 



def order_trace(det, badpix, poly_order=2, sigma_threshold=3, sub_factor=64, order_length_min=125, debug=False):
    im = np.where(badpix, signal.medfilt2d(det, 7), det)
    xx = np.arange(im.shape[1])
    xx_bin = np.arange((sub_factor-1)/2., im.shape[1], sub_factor)
    # xx_bin = xx[::sub_factor]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        im_bin = np.nanmedian(im.reshape(im.shape[0], im.shape[1]//sub_factor, sub_factor), axis=2)
    xx_loc, upper, lower  = [], [], []
    poly_upper, poly_lower  = [], []

    if debug:
        plt.imshow(im, vmin=1000, vmax=2e4)

    im_grad = np.abs(im_bin[1:,:]-im_bin[:-1,:])
    cont_std = np.nanstd(im_grad, axis=0)
    im_grad = im_grad - cont_std*sigma_threshold
    im_grad[im_grad<0] = 0

    width = 3
    len_threshold = 80
    for i in range(im_grad.shape[1]):
        yy = im_grad[:,int(i)]
        # indices,_ = signal.find_peaks(im_grad[:,i], distance=30, height=cont_std[i]*sigma_threshold)
        indices = signal.argrelextrema(yy, np.greater)[0]
        if len(indices)%2!=0 or (indices[1::2]-indices[::2]).min() < len_threshold:
            print("Warning: Data are too noisy to clearly identify edges. Consider increase 'sub_factor' in 'order_trace'.")
            continue
        cens = np.array([np.sum(xx[int(p-width):int(p+width)]*yy[int(p-width):int(p+width)])/np.sum(yy[int(p-width):int(p+width)]) for p in indices])
        lower.append(cens[::2])
        upper.append(cens[1::2])
        xx_loc.append(xx_bin[i])

        if len(indices)!=14:
            print(len(indices))
            for (k1, k2) in zip(indices,cens):
                plt.axvline(k1, color='g')
                plt.axvline(k2, color='r')
            plt.plot(im_grad[:,i])
            # plt.axhline(sigma_threshold*cont_std[i])
            plt.show()
            # raise RuntimeError("Traces are not identified correctly.")

    upper = np.array(upper).T
    lower = np.array(lower).T
    for (loc_up, loc_low) in zip(upper, lower):
        poly_up = Poly.polyfit(xx_loc, loc_up, poly_order)
        poly_low = Poly.polyfit(xx_loc, loc_low, poly_order)
        yy_up = Poly.polyval(xx, poly_up)
        yy_low = Poly.polyval(xx, poly_low)
        slit_len = yy_up - yy_low
        if slit_len.min() > order_length_min:
            poly_upper.append(poly_up)
            poly_lower.append(poly_low)
            #if the width of order varies a lot, it likely resides at the edge or not ptoperly tarced. Then try to fix the width.
        # elif (slit_len.max()-slit_len.min())>2:
            #identify whether the upper or lower trace hits the edge.
            # TODO
        if debug:
            print(slit_len.max()-slit_len.min())
            plt.plot(xx, yy_up, 'r')
            plt.plot(xx, yy_low, 'r')
            # plt.plot(xx, slit_len)
            # plt.show()

    # # test against pipeline
    # tw_filename = '/data2/yzhang/SupJup/2M0103/cr2res_util_calib_calibrated_collapsed_tw.fits'
    # x = np.arange(1, 2049)
    # tw = fits.getdata(tw_filename, 1)
    # for order in range(2, 9):
    #     p_upper = tw['Upper'][tw['Order']==order][0]
    #     y_upper = Poly.polyval(x, p_upper) 
    #     p_lower = tw['Lower'][tw['Order']==order][0]
    #     y_lower = Poly.polyval(x, p_lower) 
    #     p_wave = tw['Wavelength'][tw['Order']==order][0]
    #     wave = Poly.polyval(x, p_wave)
    #     # plt.plot(x-1., y_upper-1., 'blue')
    #     # plt.plot(x-1., y_lower-1., 'blue')
    #     slit_len = y_upper - y_lower
    #     plt.plot(x-1, slit_len)
    #     plt.show()

    if debug:
        print("%.d orders identified" % len(poly_upper))
        plt.show()

    return [poly_upper, poly_lower]

def util_order_trace(im, bpm, debug=False):
    poly_trace  = []
    for d, (det, badpix) in enumerate(zip(im, bpm)):
        trace= order_trace(det, badpix, debug=debug)
        poly_trace.append(trace)
    return poly_trace

def slit_curve(det, badpix, trace, poly_order=2, spacing=40, sub_factor=4, debug=False):
    badpix = (badpix | np.isnan(det))
    im = np.where(badpix, signal.medfilt2d(det, 5), det)
    # im = np.where(np.isnan(im), signal.medfilt2d(im, 5), im)
    im = np.nan_to_num(im)
    xx = np.arange(im.shape[1])
    trace_upper, trace_lower = trace
    meta0, meta1, meta2 = [], [], [] # (order, poly_meta) 
    for o, (upper, lower) in enumerate(zip(trace_upper, trace_lower)):
        # if debug:
        #     plt.imshow(im, vmin=200, vmax=1e4)
        middle = (upper+lower)/2.
        yy_upper = Poly.polyval(xx, upper)
        yy_lower = Poly.polyval(xx, lower)
        yy_mid = Poly.polyval(xx, middle)
        if debug:
            plt.plot(xx, yy_upper, 'r')
            plt.plot(xx, yy_lower, 'r')
        slit_image, x_slit, poly_slit = [], [], []
        for row in range(int(yy_lower.min()), int(yy_upper.max()), sub_factor):
            peaks, properties = signal.find_peaks(im[row], distance=spacing, width=5)
            width = np.median(properties['widths'])
            peaks = peaks[(peaks<(im.shape[1]-width)) & (peaks>(width))]
            cens = [np.sum(xx[int(p-width):int(p+width)]*im[row][int(p-width):int(p+width)])/np.sum(im[row][int(p-width):int(p+width)]) for p in peaks]
            slit_image.extend([[p, row] for p in cens])
            # print(np.diff(cens))
            if np.abs(row-int(yy_mid[len(xx)//2])) < sub_factor:
                pivot = cens
                bins = sorted([x-spacing/2. for x in cens] + [x+spacing/2. for x in cens])
            # print(len(peaks))
            # plt.plot(xx, im[row])
            # plt.plot(peaks, im[row][peaks],'x')
            # plt.plot(cens, im[row][peaks],'x')
            # # for p in peaks:
            # #     plt.axvline(p-width)
            # #     plt.axvline(p+width)
            # plt.show()
        slit_image = np.array(slit_image)
        indices = np.digitize(slit_image[:,0], bins)
        for i in range(1, len(bins), 2):
            xs = slit_image[:,0][indices == i]
            ys = slit_image[:,1][indices == i]
            poly = Poly.polyfit(ys, xs, poly_order)
            poly_orth = Poly.polyfit(xs, ys, poly_order)

            ### find mid point on slit image, i.e. the intersection of two polynomials
            root = Poly.polyroots(poly_orth-middle)
            root = root[(root>int(xs.min()-2))&(root<int(xs.max()+2))]
            x_slit.append(root.mean())
            poly_slit.append(poly)
            # plt.plot(root.mean(), Poly.polyval(root.mean(), middle), 'ko')
            yy = np.arange(int(yy_lower.min()-5), int(yy_upper.max()+5))
            # plt.plot(Poly.polyval(yy, poly), yy, 'r')
        poly_slit = np.array(poly_slit)
        poly_meta0 = Poly.polyfit(x_slit, poly_slit[:,0], poly_order)
        poly_meta1 = Poly.polyfit(x_slit, poly_slit[:,1], poly_order)
        poly_meta2 = Poly.polyfit(x_slit, poly_slit[:,2], poly_order)
        meta0.append(poly_meta0)
        meta1.append(poly_meta1)
        meta2.append(poly_meta2)

        if debug:
            xx_grid = pivot
            # xx_grid = np.arange(0, im.shape[1], 100)
            poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), Poly.polyval(xx_grid, poly_meta1),Poly.polyval(xx_grid, poly_meta2)]).T
            for x in range(len(xx_grid)):
                plt.plot(Poly.polyval(yy, poly_full[x]), yy, 'r:', zorder=10)
    # if debug:
    #     plt.show()

    # check rectified image
    if debug:
        spectral_rectify_interp(im, trace, [meta0, meta1, meta2], debug=debug)
    return [meta0, meta1, meta2]

def spectral_rectify_interp(im, trace, slit_mata, debug=False):
    im_rect_spec = np.copy(im)
    xx_grid = np.arange(0, im.shape[1])
    meta0, meta1, meta2 = slit_mata
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower, poly_meta0, poly_meta1, poly_meta2) in enumerate(zip(trace_upper, trace_lower, meta0, meta1, meta2)):
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), Poly.polyval(xx_grid, poly_meta1),Poly.polyval(xx_grid, poly_meta2)]).T
        isowlen_grid = np.empty((len(yy_grid), len(xx_grid)))
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])
                                
        for i, (x_isowlen, data_row) in enumerate(zip(isowlen_grid, im_rect_spec[yy_grid.astype(int)])):
            mask = np.isnan(data_row)
            if np.sum(mask)>0.5*len(mask):
                continue                      
            im_rect_spec[int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], data_row[~mask], kind='cubic', bounds_error=False, fill_value=np.nan)(x_isowlen)
    if debug:
        plt.imshow(im_rect_spec, vmin=200, vmax=1e4)
        plt.show()
    return im_rect_spec

def mean_collapse(im, trace, slit_mata, f0=0.5, fw=0.5, sigma=5, debug=False):
    im_copy = np.copy(im)
    xx_grid = np.arange(0, im.shape[1])
    blaze_orders = []
    meta0, meta1, meta2 = slit_mata
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower, poly_meta0, poly_meta1, poly_meta2) in enumerate(zip(trace_upper, trace_lower, meta0, meta1, meta2)):
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        blaze = np.zeros_like(xx_grid)
        indices = [range(int(low+(f0-fw)*(up-low)), int(low+(f0+fw)*(up-low))+1) for (up, low) in zip(yy_upper, yy_lower)]
        for i, indice in enumerate(indices):
            blaze[i],_, _ = stats.sigma_clipped_stats(im_copy[indice,i], sigma=sigma)
        plt.plot(blaze)
        plt.show()
        blaze_orders.append(blaze)
        # poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), Poly.polyval(xx_grid, poly_meta1),Poly.polyval(xx_grid, poly_meta2)]).T

    return np.array(blaze_orders)

def blaze_norm(im, trace, slit_mata, blaze_orders, f0=0.5, fw=0.5, debug=False):
    im_norm = np.copy(im)
    im_copy = np.ones_like(im)
    xx_grid = np.arange(0, im.shape[1])
    meta0, meta1, meta2 = slit_mata
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower, poly_meta0, poly_meta1, poly_meta2, blaze) in enumerate(zip(trace_upper, trace_lower, meta0, meta1, meta2, blaze_orders)):
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), Poly.polyval(xx_grid, poly_meta1),Poly.polyval(xx_grid, poly_meta2)]).T
        isowlen_grid = np.empty((len(yy_grid), len(xx_grid)))
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])
                                
        for i, x_isowlen in enumerate(isowlen_grid):
            mask = np.isnan(blaze)
            if np.sum(mask)>0.5*len(mask):
                continue                      
            im_copy[int(yy_grid[0]+i)] = interp1d(x_isowlen[~mask], blaze[~mask], kind='cubic', bounds_error=False, fill_value=np.nan)(xx_grid)
    im_norm /= im_copy
    if debug:
        plt.imshow(im_copy, vmin=200, vmax=1e4)
        plt.show()
        plt.imshow(im_norm, vmin=0.8, vmax=1.2)
        plt.show()

        # indices = [range(int(low+(f0-fw)*(up-low)), int(low+(f0+fw)*(up-low))+1) for (up, low) in zip(yy_upper, yy_lower)]
        # for i, indice in enumerate(indices):
        #     blaze[i],_, _ = stats.sigma_clipped_stats(im_copy[indice,i], sigma=sigma)
        # plt.plot(blaze)
        # plt.show()
        # blaze_orders.append(blaze)
        # poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), Poly.polyval(xx_grid, poly_meta1),Poly.polyval(xx_grid, poly_meta2)]).T

    return im_norm



def util_slit_curve(im, bpm, tw, debug=False):
    slit_meta = [] # (detector, poly_meta2, order) 
    for d, (det, badpix, trace) in enumerate(zip(im, bpm, tw)):
        slit_meta.append(slit_curve(det, badpix, trace, debug=debug))
    return slit_meta

    

def util_extract_blaze(flat, bpm, tw, slit, debug=True):
    blaze_det = [] 
    for d, (det, badpix, trace, slit_meta) in enumerate(zip(flat, bpm, tw, slit)):
        det[badpix] = np.nan
        trace_upper, trace_lower = trace
        meta0, meta1, meta2 = slit_meta
        xx_grid = np.arange(det.shape[1])
        det_rect = spectral_rectify_interp(det, trace, slit_meta, debug=debug)
        blaze_orders = mean_collapse(det_rect, trace, slit_meta, debug=debug)
        s = blaze_norm(det, trace, slit_meta, blaze_orders, debug=debug)

        blaze_det.append(blaze_orders)

    return blaze_det



#-------------------------------------------------------------------------

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
