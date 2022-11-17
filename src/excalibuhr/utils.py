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


def wfits(fname, im, hdr=None, im_err=None):
    """wfits - write image to file fname, 
    overwriting any old file"""
    if im_err is None:
        primary_hdu = fits.PrimaryHDU(im, header=hdr)
        new_hdul = fits.HDUList([primary_hdu])
    else:
        primary_hdu = fits.PrimaryHDU(im, header=hdr)
        hdu = fits.PrimaryHDU(im_err, header=hdr)
        new_hdul = fits.HDUList([primary_hdu, hdu])
    new_hdul.writeto(fname, overwrite=True, output_verify='ignore') 

def util_master_dark(dt, collapse='median', badpix_clip=5):
    # Combine the darks
    if collapse == 'median':
        master = np.nanmedian(dt, axis=0)
    elif collapse == 'mean':
        master = np.nanmean(dt, axis=0)
    
    # Calculate the read-out noise as the stddev, scaled by 
    # the square-root of the number of observations
    rons = np.nanstd(dt, axis=0)/np.sqrt(len(dt))

    # Apply a sigma-clip to identify the bad pixels
    badpix = np.zeros_like(master).astype(bool)
    for i, det in enumerate(master):
        filtered_data = stats.sigma_clip(det, sigma=badpix_clip, axis=0)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
    
    return master, rons, badpix

def util_master_flat(dt, dark, collapse='median', badpix_clip=5):
    # Combine the flats
    if collapse == 'median':
        master = np.nanmedian(dt, axis=0)
    elif collapse == 'mean':
        master = np.nanmean(dt, axis=0)
    
    # Dark-subtract the master flat
    master -= dark

    # Apply a sigma-clip to identify the bad pixels
    badpix = np.zeros_like(master).astype(bool)
    for i, det in enumerate(master):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            filtered_data = stats.sigma_clip(det, sigma=badpix_clip, axis=1)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
        # plt.imshow(badpix[i])
        # plt.show()

    return master, badpix

def combine_detector_images(dt, err, collapse='mean'):
    if collapse == 'median':
        master = np.nanmedian(dt, axis=0)
        master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~np.isnan(dt), axis=0)
    elif collapse == 'mean':
        master = np.nanmean(dt, axis=0)
        master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~np.isnan(dt), axis=0)
    elif collapse == 'sum':
        master = np.nansum(dt, axis=0)
        master_err = np.sqrt(np.nansum(np.square(err), axis=0))
    # plt.imshow((master)[0], vmin=0, vmax=60)
    # plt.show()
    # plt.imshow((master_err)[0], vmin=0, vmax=30)
    # plt.show()
    return master, master_err

def detector_shotnoise(im, ron, GAIN=2., NDIT=1):
    return np.sqrt(np.abs(im)/GAIN/NDIT + ron**2)

def util_order_trace(im, debug=False):

    # Loop over each detector
    poly_trace  = []
    for d, det in enumerate(im):
        trace = order_trace(det, debug=debug)
        poly_trace.append(trace)
    return poly_trace


def util_slit_curve(im, bpm, tw, wlen_mins, wlen_maxs, debug=False):

    # Loop over each detector
    slit_meta, wlens = [], [] # (detector, poly_meta2, order) 
    for d, (det, badpix, trace, wlen_min, wlen_max) in enumerate(zip(im, bpm, tw, wlen_mins, wlen_maxs)):
        slit, wlen = slit_curve(det, badpix, trace, wlen_min, wlen_max, debug=debug)
        slit_meta.append(slit)
        wlens.append(wlen)
    return slit_meta, wlens

def util_master_flat_norm(im, bpm, tw, slit, badpix_clip_count=1e2, debug=False):
    
    # Loop over each detector
    blazes, flat_norm  = [], []
    for d, (det, badpix, trace, slit_meta) in enumerate(zip(im, bpm, tw, slit)):
        
        # Set low signal to NaN
        det[det<badpix_clip_count] = np.nan

        det_rect, _ = spectral_rectify_interp(det, np.sqrt(det), badpix, trace, slit_meta, debug=debug)

        # Retrieve the blaze function by mean-collapsing 
        # the master flat along the slit
        blaze_orders = mean_collapse(det_rect, trace, debug=debug)
        blazes.append(blaze_orders)

        flat_norm.append(blaze_norm(det, trace, slit_meta, blaze_orders, debug=debug))

    return flat_norm, blazes


    
def util_correct_readout_artifact(im, err, bpm, tw, debug=False):
    im_cor, err_cor  = [],[]
    for d, (det, det_err, badpix, trace) in enumerate(zip(im, err, bpm, tw)):
        det_cor, det_err_cor = readout_artifact(det, det_err, badpix, trace, debug=debug)
        im_cor.append(det_cor)
        err_cor.append(det_err_cor)
    return np.array(im_cor), np.array(err_cor)

def util_flat_fielding(im, im_err, flat, debug=False):
    im_corr = np.copy(im)
    err_corr = np.copy(im_err)
    for d, (det, err, f) in enumerate(zip(im, im_err, flat)):
        badpix = np.isnan(f).astype(bool)
        im_corr[d][~badpix] = det[~badpix]/f[~badpix]
        err_corr[d][~badpix] = err[~badpix]/f[~badpix]
    if debug:
        plt.imshow(im_corr[0], vmin=-20, vmax=20)
        plt.show()
        benchmark = fits.getdata('/mnt/media/data/Users/yzhang/Projects/2M0103_CRIRES/2021-10-16/product/obs_nodding/cr2res_obs_nodding_combinedB_000.fits', 2)
        plt.imshow(err_corr[0], vmin=0, vmax=10)
        plt.show()
        plt.imshow(benchmark, vmin=0, vmax=10)
        plt.show()
    return im_corr, err_corr

def util_extract_spec(im, im_err, bpm, tw, slit, blazes, gains, f0=0.5, aper_half=15, debug=False):
    flux, err  = [], []
    for d, (det, det_err, badpix, trace, slit_meta, blaze, gain) in enumerate(zip(im, im_err,  bpm, tw, slit, blazes, gains)):
        det_rect, err_rect = spectral_rectify_interp(det, det_err,  badpix, trace, slit_meta, debug=False)
        # det_rect, err_rect = trace_rectify_interp(det_rect, err_rect, trace, debug=False)
        f_opt, f_err = extract_spec(det_rect, err_rect, badpix, trace, gain=gain, f0=f0, aper_half=aper_half, debug=debug)
        flux.append(f_opt)
        err.append(f_err)

    return np.array(flux), np.array(err)

def util_wlen_solution(dt, dt_err, wlen_init, blazes, debug=False):
    flux, err  = [], []
    # w_tellu, f_tellu = run_sky_calc()
    for d, (flux, flux_err, w_init, blaze) in enumerate(zip(dt, dt_err, wlen_init, blazes)):
        w_cal = wlen_solution(flux, flux_err, w_init, blaze, debug=False)
        # flux.append(f_opt)
        # err.append(f_err)
    return np.array(flux), np.array(err)

def order_trace(det, poly_order=2, sigma_threshold=3, sub_factor=64, order_length_min=125, debug=False):

    # Replace NaN-pixels with the median-filtered values
    # im = np.copy(det)
    im = np.where(np.isnan(det), signal.medfilt2d(det, 5), det)
    # im = np.where(np.isnan(im), signal.medfilt2d(im, 5), im)
    # im = np.nan_to_num(im)

    # Sub-sample the image along the horizontal axis
    xx = np.arange(im.shape[1])
    xx_bin = np.arange((sub_factor-1)/2., im.shape[1], sub_factor)
    # xx_bin = xx[::sub_factor]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        im_bin = np.nanmedian(im.reshape(im.shape[0], 
                                         im.shape[1]//sub_factor, 
                                         sub_factor), 
                              axis=2)
    
    xx_loc, upper, lower   = [], [], []
    poly_upper, poly_lower = [], []

    if debug:
        plt.imshow(im, vmin=1000, vmax=2e4)

    # Subtract a shifted image from its un-shifted self
    # The result is approximately the trace edge
    im_grad = np.abs(im_bin[1:,:]-im_bin[:-1,:])

    # Set insignificant signal to 0
    cont_std = np.nanstd(im_grad, axis=0)
    im_grad[(im_grad < cont_std*sigma_threshold)] = 0

    width = 3
    len_threshold = 80
    # Loop over each column in the sub-sampled image
    for i in range(im_grad.shape[1]):
        yy = im_grad[:,int(i)]
        # indices,_ = signal.find_peaks(im_grad[:,i], distance=30, height=cont_std[i]*sigma_threshold)
        
        # Find the pixels where the signal is significant
        indices = signal.argrelextrema(yy, np.greater)[0]

        if len(indices)%2!=0 or (indices[1::2]-indices[::2]).min() < len_threshold:
            print("Warning: Data are too noisy to clearly identify edges. Consider increase 'sub_factor' in 'order_trace'.")
            continue

        # Find the y-coordinates of the edges, weighted by the 
        # significance of the signal (i.e. center-of-mass)
        cens = np.array([np.sum(xx[int(p-width):int(p+width)]*yy[int(p-width):int(p+width)]) / \
                         np.sum(yy[int(p-width):int(p+width)]) \
                         for p in indices])

        # Order of y-coordinates is lower, upper, lower, ...
        lower.append(cens[::2])
        upper.append(cens[1::2])

        # Real x-coordinate of this column
        xx_loc.append(xx_bin[i])

        # if len(indices)!=14:
        #     print(len(indices))
        #     for (k1, k2) in zip(indices,cens):
        #         plt.axvline(k1, color='g')
        #         plt.axvline(k2, color='r')
        #     plt.plot(im_grad[:,i])
        #     # plt.axhline(sigma_threshold*cont_std[i])
        #     plt.show()
        #     # raise RuntimeError("Traces are not identified correctly.")

    upper = np.array(upper).T
    lower = np.array(lower).T
    # Loop over each order
    for (loc_up, loc_low) in zip(upper, lower):
        # Fit polynomials to the upper and lower edges of each order
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
            # print(slit_len.max()-slit_len.min()) #whether the slit length changes across dispersion axis
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



def slit_curve(det, badpix, trace, wlen_min, wlen_max, poly_order=2, spacing=40, sub_factor=4, debug=False):

    # Replace bad pixels with the median-filtered values
    badpix = (badpix | np.isnan(det))
    im = np.where(badpix, signal.medfilt2d(det, 5), det)
    im = np.where(np.isnan(im), signal.medfilt2d(im, 5), im)
    im = np.nan_to_num(im)

    xx = np.arange(im.shape[1])
    trace_upper, trace_lower = trace
    meta0, meta1, meta2, wlen = [], [], [], [] # (order, poly_meta) 
    # if debug:
    #     plt.imshow(im, vmin=200, vmax=1e4)
    
    # Loop over each order
    for o, (upper, lower, w_min, w_max) in enumerate(zip(trace_upper, trace_lower, wlen_min, wlen_max)):
        # Find the upper, central, and lower edges of the order 
        # with the polynomial coefficients
        middle = (upper+lower)/2.
        yy_upper = Poly.polyval(xx, upper)
        yy_lower = Poly.polyval(xx, lower)
        yy_mid = Poly.polyval(xx, middle)

        if debug:
            plt.plot(xx, yy_upper, 'r')
            plt.plot(xx, yy_lower, 'r')
        
        # Loop over each pixel-row in a sub-sampled image
        slit_image, x_slit, poly_slit = [], [], []
        for row in range(int(yy_lower.min()), int(yy_upper.max()), sub_factor):
            # Find the pixels (along horizontal axis) where signal is significant
            peaks, properties = signal.find_peaks(im[row], distance=spacing, width=5) # doesn't work with nans
            width = np.median(properties['widths'])
            peaks = peaks[(peaks<(im.shape[1]-width)) & (peaks>(width))]

            # Calculate center-of-mass of the peaks
            cens = [np.sum(xx[int(p-width):int(p+width)]*im[row][int(p-width):int(p+width)]) / \
                    np.sum(im[row][int(p-width):int(p+width)]) \
                    for p in peaks]
            slit_image.extend([[p, row] for p in cens])
            # print(np.diff(cens))

            # Select the rows that are closest to the mid-point
            if np.abs(row-int(yy_mid[len(xx)//2])) < sub_factor:
                pivot = cens
                bins = sorted([x-spacing/2. for x in cens] + \
                              [x+spacing/2. for x in cens])

            """
            print(len(peaks))
            plt.plot(xx, im[row])
            plt.plot(peaks, im[row][peaks],'x')
            plt.plot(cens, im[row][peaks],'x')
            # for p in peaks:
            #     plt.axvline(p-width)
            #     plt.axvline(p+width)
            plt.show()
            """
        
        #print(slit_image)

        slit_image = np.array(slit_image)
        # Index of bin to which each peak belongs
        indices = np.digitize(slit_image[:,0], bins)

        # Loop over every other bin
        for i in range(1, len(bins), 2):
            xs = slit_image[:,0][indices == i] # x-coordinate of peaks
            ys = slit_image[:,1][indices == i] # y-coordinate of corresponding rows

            # Fit a polynomial to the the fpet signal
            poly = Poly.polyfit(ys, xs, poly_order)
            poly_orth = Poly.polyfit(xs, ys, poly_order)

            # Find mid-point on slit image, i.e. the intersection of two polynomials
            root = Poly.polyroots(poly_orth-middle)
            if np.iscomplexobj(root):
                continue
            # Select the intersection within the valid x-coordinates
            root = root[(root>int(xs.min()-2)) & (root<int(xs.max()+2))]
            x_slit.append(root.mean())

            poly_slit.append(poly)
            #plt.plot(root.mean(), Poly.polyval(root.mean(), middle), 'ko')
            yy = np.arange(int(yy_lower.min()-5), int(yy_upper.max()+5))
            #plt.plot(Poly.polyval(yy, poly), yy, 'r')

        # Fit a polynomial to the polynomial coefficients
        # using the x-coordinates of the fpet signal
        poly_slit = np.array(poly_slit)
        poly_meta0 = Poly.polyfit(x_slit, poly_slit[:,0], poly_order)
        poly_meta1 = Poly.polyfit(x_slit, poly_slit[:,1], poly_order)
        poly_meta2 = Poly.polyfit(x_slit, poly_slit[:,2], poly_order)
        meta0.append(poly_meta0)
        meta1.append(poly_meta1)
        meta2.append(poly_meta2)
        
        # Determine the wavelength solution along the detectors/orders
        ww = np.linspace(w_min, w_max, len(xx))
        grid_poly = Poly.polyfit([0, len(xx)], [w_min, w_max], 1) # Wavelength grid
        # Mapping from measured positions to linear spacing 
        w_lin = np.linspace(x_slit[0], x_slit[-1], len(x_slit))
        cal_poly = Poly.polyfit(w_lin, x_slit-w_lin, 2)
        xx_offset = Poly.polyval(xx, cal_poly) # relative shift in pixel
        ww_cal = Poly.polyval(xx+xx_offset, grid_poly) # to wavelength
        wlen.append(ww_cal)

        
        if debug:
            xx_grid = pivot
            # xx_grid = np.arange(0, im.shape[1], 100)
            poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), 
                                  Poly.polyval(xx_grid, poly_meta1),
                                  Poly.polyval(xx_grid, poly_meta2)]).T
            for x in range(len(xx_grid)):
                plt.plot(Poly.polyval(yy, poly_full[x]), yy, 'r--', zorder=10)

    if debug:
        plt.imshow(np.log10(im), )#vmin=200, vmax=2e4)
        plt.show()
        # plt.plot(w_lin, x_slit-w_lin)
        # plt.plot(w_lin, Poly.polyval(w_lin, cal_poly))
        # plt.show()
        # plt.plot(xx, ww_cal-ww)
        # plt.show()

    # check rectified image
    # if debug:
    #     spectral_rectify_interp(im, trace, [meta0, meta1, meta2], debug=debug)
    return [meta0, meta1, meta2], wlen

def spectral_rectify_interp(im, im_err, badpix, trace, slit_meta, debug=False):

    bpm = (badpix.astype(bool) | np.isnan(im))
    im_rect_spec = np.copy(im)
    err_rect_spec = np.copy(im_err)

    # im_rect_spec[badpix.astype(bool)] = np.nan
    # if debug:
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(im[1095:1285, 1040:1230], vmin=-20, vmax=20)
    #     plt.savefig('im.png')
    #     plt.show()
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(im_rect_spec[1095:1285, 1040:1230], vmin=-20, vmax=20)
    #     plt.savefig('im_b.png')
    #     plt.show()

    bpm_interp = np.zeros_like(bpm).astype(float)
    xx_grid = np.arange(0, im.shape[1])
    # xx_grid = np.arange(0.-edge, im.shape[1]+edge)
    meta0, meta1, meta2 = slit_meta
    trace_upper, trace_lower = trace

    # Loop over each order
    for o, (poly_upper, poly_lower, poly_meta0, poly_meta1, poly_meta2) in \
        enumerate(zip(trace_upper, trace_lower, meta0, meta1, meta2)):

        # Get the upper and lower edges of the order
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        # Retrieve a pixel-grid using the slit-curvature 
        # polynomial coefficients
        poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), 
                              Poly.polyval(xx_grid, poly_meta1), 
                              Poly.polyval(xx_grid, poly_meta2)]).T
        isowlen_grid = np.empty((len(yy_grid), len(xx_grid)))
        # Loop over the horizontal axis
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])

        # Loop over each row in the order
        for i, (x_isowlen, data_row, err_row, mask) in \
            enumerate(zip(isowlen_grid, im[yy_grid], im_err[yy_grid], bpm[yy_grid])):

            # mask = np.isnan(data_row)
            if np.sum(mask)>0.5*len(mask):
                continue

            # Correct for the slit-curvature by interpolating onto the pixel-grid
            im_rect_spec[int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], data_row[~mask], kind='cubic', 
                                                       bounds_error=False, fill_value=np.nan
                                                       )(x_isowlen)

            err_rect_spec[int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], err_row[~mask], kind='cubic', 
                                                        bounds_error=False, fill_value=np.nan
                                                        )(x_isowlen)

            # Correct for the slit tilt? ...
            # ...

    #         bpm_interp[int(yy_grid[0]+i)] = interp1d(xx_grid, mask.astype(float), kind='cubic', bounds_error=False, fill_value=np.nan)(x_isowlen)
    # badpix_new = np.abs(bpm_interp)>1e-1
    if debug:
        plt.imshow(im_rect_spec, )#vmin=-20, vmax=20)
        plt.show()

    return im_rect_spec, err_rect_spec

def trace_rectify_interp(im, im_err, trace, debug=False):
    im_copy = np.copy(im)
    err_copy = np.copy(im_err)
    xx_grid = np.arange(0, im.shape[1])
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower) in enumerate(zip(trace_upper, trace_lower)):
        poly_mid = (poly_upper + poly_lower)/2.
        yy_mid = Poly.polyval(xx_grid, poly_mid)
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        shifts = yy_mid - yy_mid[len(xx_grid)//2] 
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        for x in xx_grid:
            mask = np.isnan(im[yy_grid,x]).astype(bool)
            if np.sum(mask)>0.5*len(mask):
                continue 
            im_copy[yy_grid,x] = interp1d((yy_grid)[~mask], im[yy_grid,x][~mask], kind='cubic', bounds_error=False, fill_value='extrapolate')(yy_grid+shifts[x])
            err_copy[yy_grid,x] = interp1d((yy_grid)[~mask], im_err[yy_grid,x][~mask], kind='cubic', bounds_error=False, fill_value='extrapolate')(yy_grid+shifts[x])
    if debug:
        plt.imshow(im_copy, vmin=-20, vmax=20)
        plt.show()
    return im_copy, err_copy

def mean_collapse(im, trace, f0=0.5, fw=0.5, sigma=5, edge=10, debug=False):
    im_copy = np.copy(im)
    xx_grid = np.arange(0, im.shape[1])
    # xx_grid = np.arange(0.-edge, 2048+edge)
    blaze_orders = []
    trace_upper, trace_lower = trace

    # Loop over each order
    for o, (poly_upper, poly_lower) in enumerate(zip(trace_upper, trace_lower)):
        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        blaze = np.zeros_like(xx_grid, dtype=np.float64)
        indices = [range(int(low+(f0-fw)*(up-low)), 
                         int(low+(f0+fw)*(up-low))+1) \
                   for (up, low) in zip(yy_upper, yy_lower)]

        # Loop over each column
        for i in range(len(indices)):
            mask = np.isnan(im_copy[indices[i],i])
            if np.sum(mask)>0.9*len(mask):
                blaze[i] = np.nan
            else:
                blaze[i], _, _ = stats.sigma_clipped_stats(im_copy[indices[i],i][~mask], 
                                                           sigma=sigma)
        blaze_orders.append(blaze)
        # plt.plot(xx_grid, blaze)
        # plt.show()

    return np.array(blaze_orders)

def blaze_norm(im, trace, slit_meta, blaze_orders, edge=10, f0=0.5, fw=0.5, debug=False):
    
    xx_grid_o = np.arange(0, im.shape[1])
    im_norm = np.copy(im)
    im_copy = np.ones_like(im, dtype=np.float64)*np.nan
    xx_grid = np.arange(0, im.shape[1])
    # xx_grid = np.arange(0.-edge, 2048+edge)
    meta0, meta1, meta2 = slit_meta
    trace_upper, trace_lower = trace

    print('blaze_orders', blaze_orders.shape)

    #plt.plot(xx_grid, blaze_orders[0])
    #plt.show()

    # Loop over each order
    for o, (poly_upper, poly_lower, poly_meta0, poly_meta1, poly_meta2, blaze) in \
        enumerate(zip(trace_upper, trace_lower, meta0, meta1, meta2, blaze_orders)):

        yy_upper = Poly.polyval(xx_grid, poly_upper)
        yy_lower = Poly.polyval(xx_grid, poly_lower)
        # yy_mid = (yy_upper+yy_lower)/2.
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        # Retrieve a pixel-grid using the slit-curvature 
        # polynomial coefficients
        poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), 
                              Poly.polyval(xx_grid, poly_meta1),
                              Poly.polyval(xx_grid, poly_meta2)]).T
        isowlen_grid = np.empty((len(yy_grid), len(xx_grid)))
        # Loop over the horizontal axis
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])
        
        # Loop over each row in the order
        for i, x_isowlen in enumerate(isowlen_grid):
            
            mask = np.isnan(blaze)
            if np.sum(mask)>0.9*len(mask):
                continue

            # Correct for the slit-curvature by interpolating onto the pixel-grid
            im_copy[int(yy_grid[0]+i)] = interp1d(x_isowlen[~mask], blaze[~mask], kind='cubic', 
                                                  bounds_error=False, fill_value=np.nan
                                                  )(xx_grid_o)
        # for i in range(len(xx_grid)):
        #     im_copy[int(yy_lower.min()):int(yy_lower[i])+1,i] = np.nan
        #     im_copy[int(yy_upper[i]):int(yy_upper.max()+1)+1,i] = np.nan

    # Normalize the image by the blaze function
    im_norm /= im_copy
    im_norm[im_norm<0.1] = np.nan
    if debug:
        # plt.imshow(im_copy, vmin=1e3, vmax=1.5e4)
        # plt.show()
        plt.imshow(im_norm, vmin=0.8, vmax=1.2)
        plt.show()

    return im_norm

def readout_artifact(det, det_err, badpix, trace, Nborder=10, debug=False):
    badpix = (badpix | np.isnan(det))
    im = np.copy(det)
    im_err = np.copy(det_err)
    im[badpix] = np.nan
    im_err[badpix] = np.nan

    xx = np.arange(im.shape[1])
    ron_col, err_col = np.zeros(im.shape[1]), np.zeros(im.shape[1])
    trace_upper, trace_lower = trace
    uppers, lowers = [], [] # (order, xx) 
    for o, (upper, lower) in enumerate(zip(trace_upper, trace_lower)):
        yy_upper = Poly.polyval(xx, upper)
        yy_lower = Poly.polyval(xx, lower)
        uppers.append(yy_upper)
        lowers.append(yy_lower)
        if debug:
            plt.plot(xx, yy_upper+Nborder, 'r')
            plt.plot(xx, yy_lower-Nborder, 'r')
    uppers = np.array(uppers[:-1])
    lowers = np.array(lowers[1:])
    for col in range(len(xx)):
        row = [j for i in range(len(uppers)) for j in range(int(uppers[i,col])+Nborder, int(lowers[i,col])-Nborder+1)]
        ron_col[col] = np.nanmedian(im[row,col])
        m, _, _ = stats.sigma_clipped_stats(im_err[row,col]**2)
        err_col[col] = np.sqrt(m/len(row))
    if debug:
        plt.imshow(det-ron_col, vmin=-20, vmax=20)
        plt.show()
    return det-ron_col, np.sqrt(det_err**2+err_col**2)

def extract_spec(im, im_err, bpm, trace, gain=2., f0=0.5, aper_half=20, mode='optimal', sigma=5, debug=False):
    im_copy = np.copy(im)
    bpm_copy = np.copy(bpm)
    err_copy = np.copy(im_err)
    xx_grid = np.arange(0, im.shape[1])
    flux, err = [],[]
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower) in enumerate(zip(trace_upper, trace_lower)):
        # poly_mid = (poly_upper + poly_lower)/2.
        # yy_mid = Poly.polyval(xx_grid, poly_mid)
        yy_upper = Poly.polyval(xx_grid, poly_upper)[len(xx_grid)//2] 
        yy_lower = Poly.polyval(xx_grid, poly_lower)[len(xx_grid)//2] 
        slit_len = (yy_upper-yy_lower)
        im_sub = im_copy[int(yy_lower):int(yy_upper+1)]
        im_err_sub = err_copy[int(yy_lower):int(yy_upper+1)]
        bpm_sub = bpm_copy[int(yy_lower):int(yy_upper+1)]
        obj_cen = slit_len*f0
        # print(slit_len, f0, obj_cen)
        
        f_opt, f_err = optimal_extraction(im_sub.T, im_err_sub.T**2, bpm_sub.T, int(np.round(obj_cen)), aper_half, gain=gain, debug=debug) 
        flux.append(f_opt)
        err.append(f_err)

    return np.array(flux), np.array(err)


def optimal_extraction(D_full, V_full, bpm_full, obj_cen, aper_half, return_profile=False, badpix_clip=3, max_iter=10, gain=2., NDIT=1., etol=1e-6, debug=False):
    # TODO: NDIT from header.
    D = D_full[:,obj_cen-aper_half:obj_cen+aper_half+1]
    V = V_full[:,obj_cen-aper_half:obj_cen+aper_half+1]
    bpm = bpm_full[:,obj_cen-aper_half:obj_cen+aper_half+1]
    filtered_D = stats.sigma_clip(D, sigma=badpix_clip, axis=0)
    bpm = filtered_D.mask | bpm 
    D = np.nan_to_num(D, nan=etol)
    V = np.nan_to_num(V, nan=1./etol)
    
    M_bp_init = ~bpm.astype(bool)
    wave_x = np.arange(D.shape[0])
    spatial_x = np.arange(D.shape[1])
    f_std = np.nansum(D*M_bp_init.astype(float), axis=1)
    D_norm = np.zeros_like(D)
    P = np.zeros_like(D)

    for x in spatial_x:
        D_norm[:,x] = D[:,x]/(f_std+etol)

    for x in spatial_x:
        p_model, M_bp_new = PolyfitClip(wave_x, \
                        D_norm[:,x], 2, M_bp_init[:,x], \
                        clip=badpix_clip, plotting=False)
        P[:,x] = p_model
        # M_bp_init[:,x] = M_bp_new
    P[P<=0] = etol
    for w in wave_x:
        P[w] /= np.sum(P[w])

    ite = 0
    M_bp = np.copy(M_bp_init)
    V_new = V + D / gain / NDIT
    while ite < max_iter:
        f_opt = np.sum(M_bp*P*D/V_new, axis=1)/(np.sum(M_bp*P*P/V_new, axis=1)+etol)
        Res = M_bp * (D - P*np.tile(f_opt, (P.shape[1],1)).T)**2/V_new
        V_new = V + P*np.tile(np.abs(f_opt), (P.shape[1],1)).T /gain / NDIT
        if np.all(Res < badpix_clip**2):
            break
        for x in wave_x:
            if np.any(Res[x]>badpix_clip**2):
                M_bp[x, np.argmax(Res[x]-badpix_clip**2)] = 0.
        ite += 1 

    f_opt = np.sum(M_bp*P*D/V_new, axis=1)/(np.sum(M_bp*P*P/V_new, axis=1)+etol)
    var = np.sum(M_bp*P, axis=1)/(np.sum(M_bp*P*P/V_new, axis=1)+etol) * np.sum(Res)/(np.sum(M_bp)-len(f_opt)) # rescale errbar by chi2
    
    if debug:
        # print("Number of iterations: ", ite)
        print("Reduced chi2: ", np.sum(Res)/(np.sum(M_bp)-len(f_opt)))
        # plt.plot(f_std)
        plt.plot(f_opt)
        plt.show()
        # var = np.sum(M_bp*P, axis=1)/(np.sum(M_bp*P*P/V_new, axis=1)+etol) 
        # plt.plot(f_opt/np.sqrt(var))
        plt.plot(f_opt/np.sqrt(var))
        plt.show()


    if return_profile:
        return f_opt, np.sqrt(var), P, M_bp
    else:
        return f_opt, np.sqrt(var)


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

def PolyfitClip(x, y, dg, m, w=None, clip=4., max_iter=10, \
                plotting=False):
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
    if np.sum(m) < 0.1*len(m):
        return np.zeros(dg), m
    xx = np.copy(x)
    yy = np.copy(y)
    mask = np.copy(m)
    if w is None:
        ww = np.ones_like(xx)
    else:
        ww = np.copy(w)
    # mask = (np.isnan(xx)) | (np.isnan(yy)) | (np.isinf(xx)) | (np.isinf(yy))
    ite=0
    while ite < max_iter:
        poly = Poly.polyfit(xx[mask], yy[mask], dg, w=ww[mask])
        y_model = Poly.polyval(xx, poly)
        res = yy - y_model
        threshold = np.std(res[mask])*clip
        if plotting and ite>0:
            # plt.plot(yy)
            # plt.plot(y_model)
            # plt.show()
            plt.plot(res)
            plt.axhline(threshold)
            plt.axhline(-threshold)
            plt.ylim((-1.2*threshold,1.2*threshold))
            plt.show()
        if np.any(np.abs(res[mask]) > threshold):
            mask = mask & (np.abs(res) < threshold)
        else:
            break
        ite+=1
    return y_model, mask
