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
from astropy.modeling import models, fitting
from numpy.polynomial import polynomial as Poly
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
import warnings


def wfits(fname, im, hdr=None, ext_list=None):
    """wfits - write image to file fname, 
    overwriting any old file"""
    primary_hdu = fits.PrimaryHDU(im, header=hdr)
    new_hdul = fits.HDUList([primary_hdu])
    if not ext_list is None:
        if not isinstance(ext_list, list):
            new_hdul.append(fits.PrimaryHDU(ext_list))
        else:
            for ext in ext_list:
                new_hdul.append(fits.PrimaryHDU(ext))
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

def combine_frames(dt, err, collapse='mean'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if collapse == 'median':
            master = np.nanmedian(dt, axis=0)
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~np.isnan(dt), axis=0)
        elif collapse == 'mean':
            master = np.nanmean(dt, axis=0)
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~np.isnan(dt), axis=0)
        elif collapse == 'sum':
            master = np.nansum(dt, axis=0)
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))
        elif collapse == 'weighted':
            N_frames = np.array(dt).shape
            dt_flat = np.reshape(dt, (N_frames[0], -1))
            err_flat = np.reshape(err, (N_frames[0], -1))
            # weighted by average SNR squared
            weights = np.square(np.nanmedian(dt_flat/err_flat, axis=1))
            master = np.dot(weights, dt_flat)/np.sum(weights)
            master_err = np.sqrt(np.dot(weights**2, err_flat**2))/np.sum(weights)
            master = np.reshape(master, N_frames[1:])
            master_err = np.reshape(master_err, N_frames[1:])

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


def util_slit_curve(im, tw, wlen_mins, wlen_maxs, debug=False):
    # Loop over each detector
    slit_meta, wlens = [], [] # (detector, poly_meta2, order) 
    for d, (det, trace, wlen_min, wlen_max) in enumerate(zip(im, tw, wlen_mins, wlen_maxs)):
        if d == 2:
            # fix perculiar value of the specific detector/order 
            wlen_min[-1] = 1949.093
        slit, wlen = slit_curve(det, trace, wlen_min, wlen_max, debug=debug)
        slit_meta.append(slit)
        wlens.append(wlen)
    return slit_meta, wlens

def util_master_flat_norm(im, bpm, tw, slit, badpix_clip_count=1e2, debug=False):
    
    # Loop over each detector
    blazes, flat_norm  = [], []
    for d, (det, badpix, trace, slit_meta) in enumerate(zip(im, bpm, tw, slit)):
        
        # Set low signal to NaN
        det[det<badpix_clip_count] = np.nan

        # Correct for the slit-curvature
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
        plt.imshow(err_corr[0], vmin=0, vmax=10)
        plt.show()
    return im_corr, err_corr

def util_extract_spec(im, im_err, bpm, tw, slit, blazes, gains, f0=0.5, f1=None, aper_half=20, bkg_subtract=False, f_star=None, debug=False):
    
    # Loop over each detector
    flux, err  = [], []
    for d, (det, det_err, badpix, trace, slit_meta, blaze, gain) in \
        enumerate(zip(im, im_err,  bpm, tw, slit, blazes, gains)):
        
        # Correct for the slit-curvature
        det_rect, err_rect = spectral_rectify_interp(det, det_err,  badpix, trace, slit_meta, debug=False)
        det_rect, err_rect = trace_rectify_interp(det_rect, err_rect, trace, debug=False)

        # Extract a 1D spectrum
        if bkg_subtract:
            f_opt, f_err = extract_spec(det_rect, err_rect, badpix, trace, gain=gain, \
                                    f0=f0, f1=f1, aper_half=aper_half, \
                                    bkg_subtract=bkg_subtract, \
                                    f_star=f_star[d]/blaze, debug=debug)
        else:
            f_opt, f_err = extract_spec(det_rect, err_rect, badpix, trace, gain=gain, \
                                    f0=f0, f1=f1, aper_half=aper_half, \
                                    debug=debug)
        flux.append(f_opt)
        err.append(f_err)

    return np.array(flux), np.array(err)

def util_wlen_solution(dt, wlen_init, blazes, tellu, debug=False):
    wlen_cal  = []
    for d, (flux, w_init, blaze) in enumerate(zip(dt, wlen_init, blazes)):
        w_cal = wlen_solution(flux, w_init, blaze, tellu, debug=debug)
        wlen_cal.append(w_cal)
    return np.array(wlen_cal)

def peak_slit_fraction(im, trace, debug=False):
    frac = []
    xx_grid = np.arange(im.shape[1])
    trace_upper, trace_lower = trace
    for o, (poly_upper, poly_lower) in enumerate(zip(trace_upper, trace_lower)):
        # Crop out the order from the frame
        yy_upper = Poly.polyval(xx_grid, poly_upper)[len(xx_grid)//2] 
        yy_lower = Poly.polyval(xx_grid, poly_lower)[len(xx_grid)//2] 
        slit_len = (yy_upper-yy_lower)
        im_sub = im[int(yy_lower):int(yy_upper+1)]

        # white-light (collpase along wavlengths) psf
        profile = np.nanmedian(im_sub, axis=1)
        # # Pixel-location of target
        f0 = np.argmax(profile) 
        frac.append((f0)/slit_len)

    return np.mean(frac)

def align_jitter(dt, err, pix_shift, tw, debug=False):
    dt_shift = np.copy(dt)
    err_shift = np.copy(err)
    if pix_shift < 0:
        dt_shift[:, :pix_shift, :] = dt[:, -pix_shift:, :]
        err_shift[:, :pix_shift, :] = err[:, -pix_shift:, :]
    elif pix_shift > 0:
        dt_shift[:, pix_shift:, :] = dt[:, :-pix_shift, :]
        err_shift[:, pix_shift:, :] = err[:, :-pix_shift, :]
    if debug:
        #print peak signal location
        print(pix_shift, peak_slit_fraction(dt[0], tw[0]))
    return dt_shift, err_shift

def util_unravel_spec(wlens, specs, errs, blazes=None, debug=False):
    if blazes is None:
        blazes = np.ones_like(specs)
    # shape: (wlen_settings, detectors, orders, -1)
    N_set, N_det, N_ord, Nx = wlens.shape

    # generate a evenly spaced wavelength grid 
    wlen_even = np.copy(wlens)
    wmin = wlens[:,:,:,0] 
    wmax = wlens[:,:,:,-1] 
    for i in range(N_set*N_det*N_ord):
        ind = np.unravel_index(i, (N_set, N_det, N_ord))
        wlen_even[ind] = np.linspace(wmin[ind], wmax[ind], Nx)

    wlen_flatten = np.reshape(wlens, (N_set*N_det*N_ord, Nx))
    indices = np.argsort(wlen_flatten[:,0])
    wlen = wlen_flatten[indices].flatten()
    unraveled = []
    for dt in [specs, errs, blazes, wlen_even]:
        dt_flatten = np.reshape(dt, (N_set*N_det*N_ord, Nx))
        unraveled.append(dt_flatten[indices].flatten())
    spec, err, norm, w_even = unraveled
    if debug:
        plt.plot(wlen, spec/norm)
        plt.show()

    return wlen, spec/norm, err/norm, w_even

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

    if debug:
        print("%.d orders identified" % len(poly_upper))
        plt.show()

    return [poly_upper, poly_lower]



def slit_curve(det, trace, wlen_min, wlen_max, poly_order=2, spacing=40, sub_factor=4, debug=False):

    badpix = np.isnan(det)
    im = np.ma.masked_array(det, mask=badpix)

    xx = np.arange(im.shape[1])
    trace_upper, trace_lower = trace
    meta0, meta1, meta2, wlen = [], [], [], [] # (order, poly_meta) 
    
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
        N_lines = 0
        for row in range(int(yy_lower.min()+1), int(yy_upper.max()), sub_factor):

            # combine a few rows (N=sub_factor) to increase S/N and reject outliers
            im_masked = stats.sigma_clip(im[row:row+sub_factor], sigma=2, axis=0, masked=True)
            spec = np.ma.mean(im_masked, axis=0)
            mask = (np.sum(im_masked.mask, axis=0) > 0.7*sub_factor)
            badchannel = np.argwhere(mask)[:,0]
            # mask neighboring pixels as well
            badchannel = np.concatenate((badchannel, badchannel+1, badchannel-1), axis=None)

            # measure the baseline of the spec to be 10% lowest values
            height = np.median(np.sort(spec.data[~mask])[:len(spec)//10])

            # Find the pixels (along horizontal axis) where signal is significant
            peaks, properties = signal.find_peaks(spec, distance=spacing, width=10, height=2*height) 
            # mask the peaks identified due to bad channels
            peaks = np.array([item for item in peaks if not item in badchannel])

            # leave out lines around detector edge
            width = np.median(properties['widths'])
            peaks = peaks[(peaks<(im.shape[1]-width)) & (peaks>(width))]

            # Calculate center-of-mass of the peaks
            cens = [np.sum(xx[int(p-width):int(p+width)]*spec[int(p-width):int(p+width)]) / \
                    np.sum(spec[int(p-width):int(p+width)]) \
                    for p in peaks]
            slit_image.extend([[p, row+(sub_factor-1)/2.] for p in cens])

            # generate bins to divide the peaks in groups
            # when maximum number of fpet lines are identified in the order
            if len(peaks) > N_lines:
                bins = sorted([x-spacing for x in cens] + \
                              [x+spacing for x in cens])
                N_lines = len(peaks)
            
            # plt.plot(xx, spec)
            # plt.plot(peaks, spec[peaks],'x')
            # plt.plot(cens, spec[peaks],'x')
            # # for p in badchannel:
            # #     plt.axvline(p)
            # plt.show()

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
            # x_slit stores the x-coordinates of each fpet line at the slit center
            if len(root)>0:
                x_slit.append(root.mean())
                poly_slit.append(poly)
            # plt.scatter(xs, ys)

        # Fit a polynomial to the polynomial coefficients
        # using the x-coordinates of the fpet signal
        poly_slit = np.array(poly_slit)
        poly_meta0 = Poly.polyfit(x_slit, poly_slit[:,0], poly_order)
        poly_meta1 = Poly.polyfit(x_slit, poly_slit[:,1], poly_order)
        poly_meta2 = Poly.polyfit(x_slit, poly_slit[:,2], poly_order)
        meta0.append(poly_meta0)
        meta1.append(poly_meta1)
        meta2.append(poly_meta2)
        
        if debug:
            xx_grid = x_slit
            yy = np.arange(int(yy_lower.min()-5), int(yy_upper.max()+5))
            poly_full = np.array([Poly.polyval(xx_grid, poly_meta0), 
                                  Poly.polyval(xx_grid, poly_meta1),
                                  Poly.polyval(xx_grid, poly_meta2)]).T
            for x in range(len(xx_grid)):
                plt.plot(xx_grid[x], Poly.polyval(xx_grid[x], middle), 'ko')
                plt.plot(Poly.polyval(yy, poly_full[x]), yy, 'r--', zorder=10)

        # Determine the wavelength solution along the detectors/orders
        # Mapping from measured fpet positions to a linear-spaced grid, fitting with 2nd-order poly
        ii = np.arange(len(x_slit))
        _, poly = PolyfitClip(x_slit, ii, 2)
        # # Fix the end of the wavelength grid to be values from the header
        grid_poly = Poly.polyfit([Poly.polyval(xx, poly)[0], Poly.polyval(xx, poly)[-1]], [w_min, w_max], 1)
        ww_cal = Poly.polyval(Poly.polyval(xx, poly), grid_poly)
        wlen.append(ww_cal)

        # w_slit = Poly.polyval(Poly.polyval(x_slit, poly), grid_poly)
        # plt.plot(np.diff(w_slit))
        # plt.show()

    if debug:
        plt.imshow(im, vmin=100, vmax=2e4)
        plt.show()

    # check rectified image
    # if debug:
    #     spectral_rectify_interp(im, trace, [meta0, meta1, meta2], debug=debug)
    return [meta0, meta1, meta2], wlen

def spectral_rectify_interp(im, im_err, badpix, trace, slit_meta, debug=False):

    bpm = (badpix.astype(bool) | np.isnan(im))
    im_rect_spec = np.copy(im)
    err_rect_spec = np.copy(im_err)

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

            if np.sum(mask)>0.5*len(mask):
                continue

            # Correct for the slit-curvature by interpolating onto the pixel-grid
            im_rect_spec[int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], data_row[~mask], kind='cubic', 
                                                       bounds_error=False, fill_value=np.nan
                                                       )(x_isowlen)

            err_rect_spec[int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], err_row[~mask], kind='cubic', 
                                                        bounds_error=False, fill_value=np.nan
                                                        )(x_isowlen)

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

def extract_spec(im, im_err, bpm, trace, gain=2., f0=0.5, f1=None, aper_half=20, bkg_subtract=False, f_star=None, debug=False):

    im_copy = np.copy(im)
    bpm_copy = np.copy(bpm)
    err_copy = np.copy(im_err)
    xx_grid = np.arange(0, im.shape[1])
    flux, err = [],[]
    trace_upper, trace_lower = trace

    for o, (poly_upper, poly_lower) in enumerate(zip(trace_upper, trace_lower)):
        # poly_mid = (poly_upper + poly_lower)/2.
        # yy_mid = Poly.polyval(xx_grid, poly_mid)
        
        # Crop out the order from the frame
        yy_upper = Poly.polyval(xx_grid, poly_upper)[len(xx_grid)//2] 
        yy_lower = Poly.polyval(xx_grid, poly_lower)[len(xx_grid)//2] 
        slit_len = (yy_upper-yy_lower)
        im_sub = im_copy[int(yy_lower):int(yy_upper+1)]
        im_err_sub = err_copy[int(yy_lower):int(yy_upper+1)]
        bpm_sub = bpm_copy[int(yy_lower):int(yy_upper+1)]

        # remove starlight contamination
        if bkg_subtract and f1:
            im_sub, im_err_sub = remove_starlight(im_sub, im_err_sub**2, f_star[o], slit_len*f0, slit_len*f1, gain=gain, debug=debug)

        # Pixel-location of target
        obj_cen = slit_len*f0
        if not f1 is None:
            obj_cen = slit_len*f1
        # print(slit_len, f0, obj_cen)
        
        # Extract a 1D spectrum using the optimal extraction algorithm
        f_opt, f_err = optimal_extraction(im_sub.T, im_err_sub.T**2, bpm_sub.T, int(np.round(obj_cen)), 
                                          aper_half, gain=gain, debug=debug) 
        flux.append(f_opt)
        err.append(f_err)

    return np.array(flux), np.array(err)

def remove_starlight(D_full, V_full, f_star, cen0_p, cen1_p, aper0=50, aper1=10, aper2=20, sub_factor=32, gain=2.0, NDIT=1., debug=False):
    # determine the location of peak signal from data
    spatial_x = np.arange(len(D_full))
    bkg = np.zeros_like(D_full)
    f_star = np.nan_to_num(f_star)

    polys = []
    x_sub = np.arange(0, D_full.shape[-1], sub_factor) + sub_factor/2.
    D = np.reshape(D_full, (D_full.shape[0], D_full.shape[-1]//sub_factor, sub_factor))
    D_sub = np.nanmedian(D, axis=2)

    # estimate center of the PSF peak
    # aper2: the window (half width) to determine the line center;
    #        better to be large but, not too large to avoid the companion
    profile = np.nanmedian(-D_sub, axis=1)
    p = np.argmax(profile)
    cen0_n = np.sum(spatial_x[int(p-aper2):int(p+aper2)]*profile[int(p-aper2):int(p+aper2)]) / \
                    np.sum(profile[int(p-aper2):int(p+aper2)]) 
    cen1_n = cen0_n - cen0_p + cen1_p
    profile = np.nanmedian(D_sub, axis=1)
    p = np.argmax(profile)
    cen0_p = np.sum(spatial_x[int(p-aper2):int(p+aper2)]*profile[int(p-aper2):int(p+aper2)]) / \
                    np.sum(profile[int(p-aper2):int(p+aper2)]) 

    # aper0: mask the negative primary trace
    # aper1: mask the negative secondary trace and the primary line core
    mask =  ((spatial_x>cen1_n-aper1) & (spatial_x<cen1_n+aper1)) | \
            ((spatial_x>cen0_n-aper0) & (spatial_x<cen0_n+aper0))  
    
    # not sure the secondary is located at which side
    if cen1_p - cen0_p > 0: 
        m = (~mask) & (spatial_x-cen0_p-aper1<0)
    else:
        m = (~mask) & (spatial_x-cen0_p-aper1>0)
    
    # flip PSF to the oposite side
    xx = 2.*cen0_p - spatial_x[m]
    D_sub = D_sub[m]
    # print(cen0_p, cen1_p)

    for w in range(len(x_sub)):
        poly = Poly.polyfit(xx, D_sub[:,w], 5)
        polys.append(poly)

    polys = np.array(polys)
    polys_model = np.empty((D_full.shape[-1], polys.shape[-1]))
    # suppose each polynomial coeffecient vary with wavelength linearly (1st order poly).
    for d in range(polys.shape[-1]):
        polys_model[:, d] = Poly.polyval(np.arange(D_full.shape[-1]), \
            Poly.polyfit(x_sub + sub_factor/2., polys[:, d], 1))

    indices = (spatial_x < xx[-1]) | (spatial_x> xx[0])
    for w, poly in enumerate(polys_model):
        bkg[:, w] = Poly.polyval(spatial_x, poly)
        bkg[indices] = 0.
        # plt.plot(spatial_x,  bkg[:, w])
        # plt.plot(spatial_x,  D_full[:, w])
        # plt.show()
    
    # error propogation
    V_full += bkg / gain / NDIT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Err_full = np.sqrt(V_full)

    profile_model = np.median(bkg, axis=1)

    bkg = bkg * f_star / np.nanmedian(f_star)

    if debug:
        plt.plot(profile)
        plt.plot(profile_model)
        plt.show()
        # plt.imshow(D_full, vmin=-10, vmax=20, aspect='auto')
        # plt.show()
        plt.imshow(D_full-bkg, vmin=-10, vmax=20, aspect='auto')
        plt.show()

    return D_full-bkg, Err_full



def optimal_extraction(D_full, V_full, bpm_full, obj_cen, aper_half, \
                       return_profile=False, badpix_clip=3, max_iter=10, \
                       gain=2., NDIT=1., etol=1e-6, debug=False):
    # TODO: NDIT from header.

    D = D_full[:,obj_cen-aper_half:obj_cen+aper_half+1] # Observation
    V = V_full[:,obj_cen-aper_half:obj_cen+aper_half+1] # Variance
    bpm = bpm_full[:,obj_cen-aper_half:obj_cen+aper_half+1]

    # Sigma-clip the observation and add bad-pixels to map
    filtered_D = stats.sigma_clip(D, sigma=badpix_clip, axis=0)
    bpm = filtered_D.mask | bpm 
    M_bp_init = ~bpm.astype(bool)
    D = np.nan_to_num(D, nan=etol)
    V = np.nan_to_num(V, nan=1./etol)

    wave_x = np.arange(D.shape[0])
    spatial_x = np.arange(D.shape[1])
    D_norm = np.zeros_like(D)
    f_std = np.nansum(D*M_bp_init.astype(float), axis=1)
    
    # Normalize the observation per pixel-row
    for x in spatial_x:
        D_norm[:,x] = D[:,x]/(f_std+etol)

    P = np.zeros_like(D)
    # For each row, fit polynomial, iteratively clipping pixels
    for x in spatial_x:
        p_model, _ = PolyfitClip(wave_x, \
                        D_norm[:,x], 2, M_bp_init[:,x], \
                        clip=badpix_clip, plotting=False)
        P[:,x] = p_model
        # M_bp_init[:,x] = M_bp_new
    P[P<=0] = etol


    # Normalize the polynomial model (profile) per pixel-column
    for w in wave_x:
        P[w] /= np.sum(P[w])

    ite = 0
    M_bp = np.copy(M_bp_init)
    V_new = V + D / gain / NDIT
    while ite < max_iter:
        # Optimally extracted spectrum, obtained by accounting
        # for the profile and variance
        f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1) + etol)
        
        # Residual of expanded optimally extracted spectrum and the observation
        Res = M_bp * (D - P*np.tile(f_opt, (P.shape[1],1)).T)**2/V_new

        # Calculate a new variance with the optimally extracted spectrum
        V_new = V + P*np.tile(np.abs(f_opt), (P.shape[1],1)).T / gain / NDIT

        # Halt iterations if residuals are small enough
        if np.all(Res < badpix_clip**2):
            break

        for x in wave_x:
            if np.any(Res[x]>badpix_clip**2):
                M_bp[x, np.argmax(Res[x]-badpix_clip**2)] = 0.
        ite += 1

    # Final optimally extracted spectrum
    f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1)+etol)
    # mask bad wavelength channels which contain more than 50% bad pixels
    badchannel = (np.sum(M_bp, axis=1) < len(spatial_x)*0.5)
    f_opt[badchannel] = np.nan
    # Rescale the variance by the chi2
    var = 1. / (np.sum(M_bp*P*P/V_new, axis=1)+etol) * np.sum(Res)/(np.sum(M_bp)-len(f_opt))

    if debug:
        # print("Number of iterations: ", ite)
        print("Reduced chi2: ", np.sum(Res)/(np.sum(M_bp)-len(f_opt)))
        # '''
        plt.plot(f_opt)
        plt.show()
        # plt.plot(f_opt/np.sqrt(var))
        # plt.show()
        # '''

        D[M_bp == 0] = np.nan
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].imshow(D.T, aspect='auto', vmin=np.nanmin(P*np.tile(f_opt, (P.shape[1],1)).T),
                    vmax=np.nanmax(P*np.tile(f_opt, (P.shape[1],1)).T))
        ax[1].imshow((P*np.tile(f_opt, (P.shape[1],1)).T).T, aspect='auto',
                    vmin=np.nanmin(P*np.tile(f_opt, (P.shape[1],1)).T),
                    vmax=np.nanmax(P*np.tile(f_opt, (P.shape[1],1)).T))
        plt.show()


    if return_profile:
        return f_opt, np.sqrt(var), P, M_bp
    else:
        return f_opt, np.sqrt(var)



def wlen_solution(fluxes, w_init, blazes, transm_spec, minimum_strength=0.005, debug=False):
    wlens = []

    for o in range(len(fluxes)):
        f, wlen_init, blaze = fluxes[o], w_init[o], blazes[o]
        f = f/blaze
        index_o = (transm_spec[:,0]>np.min(wlen_init)) & (transm_spec[:,0]<np.max(wlen_init))
        
        # Check if there are enough telluric features in this wavelength range
        if np.std(transm_spec[:,1][index_o]) > minimum_strength: 
            # Calculate the cross-correlation between data and template
            opt_poly = xcor_wlen_solution(f, wlen_init, transm_spec, mode='quad')
            # opt_poly, cross_corr = xcor_wlen_solution(f, wlen_init, transm_spec, mode='quad', return_cross_corr=debug)

        else:
            warnings.warn(
                    "Not enough telluric features to correct wavelength "
                    f"for order {o}"
                )
            opt_poly = [0., 1.0, 0.]

        if debug:
            print(f"{o} -> lambda: {opt_poly}")

        mean_wavel = np.mean(wlen_init)
        if len(opt_poly) == 2:
            wlen_cal = mean_wavel + opt_poly[0] + opt_poly[1] * (wlen_init - mean_wavel)
        elif len(opt_poly) == 3:
            wlen_cal = mean_wavel + opt_poly[0] + opt_poly[1] * (wlen_init - mean_wavel) \
                                  + opt_poly[2] * (wlen_init - mean_wavel)**2
        wlens.append(wlen_cal)
        if debug:
            plt.plot(wlen_init, f/np.nanmedian(f), alpha=0.8, color='b')
            plt.plot(wlen_cal, f/np.nanmedian(f), alpha=0.8, color='orange')
    if debug:
        plt.plot(transm_spec[:,0], transm_spec[:,1], alpha=0.8, color='k')
        plt.show()

    return wlens

def xcor_wlen_solution(spec, wavel, transm_spec, 
        accuracy: float = 0.02,
        offset_range: float = 0.4,
        slope_range: float = 0.01,
        c_range: float = 0.002,
        mode = 'quad',
        continuum_smooth_length : int = 101,
        return_cross_corr: bool = False)-> tuple([np.ndarray, float, float]):
        # (see pycrires: https://pycrires.readthedocs.io)
        
        template_interp = interp1d(
            transm_spec[:,0], transm_spec[:,1], 
            kind="linear", bounds_error=True
        )

        # Remove continuum and nans of spectra.
        # The continuum is estimated by smoothing the
        # spectrum with a 2nd order Savitzky-Golay filter
        spec = spec[10:-10]
        wavel = wavel[10:-10]
        nans = np.isnan(spec) + (spec<0.1*np.nanmedian(spec))
        continuum = signal.savgol_filter(
            spec[~nans], window_length=continuum_smooth_length,
            polyorder=2, mode='interp')
        
        spec_flat = spec[~nans] - continuum
        outliers = np.abs(spec_flat)>(5*np.nanstd(spec))
        spec_flat[outliers]=0

        # Don't use the edges as that sometimes gives problems
        spec_flat = spec_flat[10:-10]
        used_wavel = wavel[~nans][10:-10]

        # Prepare cross-correlation grid
        N_a = int(np.ceil(1.0 / accuracy) // 2 * 2 + 1)
        N_b = int(N_a // 3 // 2 * 2 + 1)
        N_c = N_b


        if mode == 'quad':
            a_grid = np.linspace(
                -offset_range, offset_range, N_a)[np.newaxis, np.newaxis, :, np.newaxis]
            b_grid = np.linspace(
                1.-slope_range, 1.+slope_range, N_b)[np.newaxis, :, np.newaxis, np.newaxis]
            c_grid = np.linspace(
                -c_range, c_range, N_c)[:, np.newaxis, np.newaxis, np.newaxis]

            mean_wavel = np.mean(wavel)
            wl_matrix = b_grid * (used_wavel[np.newaxis, np.newaxis, np.newaxis, :] \
                                - mean_wavel) + mean_wavel + a_grid + \
                                c_grid *(used_wavel[np.newaxis, np.newaxis, np.newaxis, :] \
                                - mean_wavel)**2
            template = template_interp(wl_matrix) - 1.

            # Calculate the cross-correlation
            # between data and template
            cross_corr = template.dot(spec_flat)

            # Find optimal wavelength solution
            opt_idx = np.unravel_index(
                np.argmax(cross_corr), cross_corr.shape)
            opt_a = a_grid[0, 0, opt_idx[2], 0]
            opt_b = b_grid[0, opt_idx[1], 0, 0]
            opt_c = c_grid[opt_idx[0], 0, 0, 0]

            if np.abs(1.-opt_b) >= slope_range or np.abs(opt_a) >= offset_range or np.abs(opt_c) >= c_range:
                warnings.warn("Hit the edge of grid when optimizing the wavelength solution."
                            f"Slope: {opt_b}({slope_range}), offset: {opt_a}({offset_range}), c: {opt_c}({c_range}).")

            if return_cross_corr:
                return  (opt_a, opt_b, opt_c), cross_corr
            else:
                return (opt_a, opt_b, opt_c)
        
        elif mode == 'linear':
            b_grid = np.linspace(
                1.-slope_range, 1.+slope_range, N_b)[:, np.newaxis, np.newaxis]
            a_grid = np.linspace(
                -offset_range, offset_range, N_a)[np.newaxis, :, np.newaxis]

            mean_wavel = np.mean(wavel)
            wl_matrix = b_grid * (used_wavel[np.newaxis, np.newaxis, :]
                                - mean_wavel) + mean_wavel + a_grid
            template = template_interp(wl_matrix) - 1.

            # Calculate the cross-correlation
            # between data and template
            cross_corr = template.dot(spec_flat)

            # Find optimal wavelength solution
            opt_idx = np.unravel_index(
                np.argmax(cross_corr), cross_corr.shape)
            opt_a = a_grid[0, opt_idx[1], 0]
            opt_b = b_grid[opt_idx[0], 0, 0]

            if np.abs(1.-opt_b) >= slope_range or np.abs(opt_a) >= offset_range:
                warnings.warn("Hit the edge of grid when optimizing the wavelength solution."
                            f"Slope: {opt_b}({slope_range}), offset: {opt_a}({offset_range}).")

            if return_cross_corr:
                return  (opt_a, opt_b), cross_corr
            else:
                return (opt_a, opt_b)



def SpecConvolve(in_wlen, in_flux, out_res, in_res=1e6, verbose=False):
    """
    Convolve the input spectrum to a lower resolution.
    ----------
    Parameters
    ----------
    in_wlen : Wavelength array 
    in_flux : spectrum at high resolution
    in_res : input resolution (high) R~w/dw
    out_res : output resolution (low)
    verbose : if True, print out the sigma of Gaussian filter used
    ----------
    Returns
    ----------
    Convolved spectrum
    """
    # delta lambda of resolution element is FWHM of the LSF's standard deviation:
    sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))

    spacing = np.mean(2.*np.diff(in_wlen)/ \
      (in_wlen[1:]+in_wlen[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF/spacing

    flux_LSF = ndimage.gaussian_filter(in_flux, \
                               sigma = sigma_LSF_gauss_filter, \
                               mode = 'nearest')
    if verbose:
        print("Guassian filter sigma = {} pix".format(sigma_LSF_gauss_filter))
    return flux_LSF

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

def PolyfitClip(x, y, dg, m=None, w=None, clip=4., max_iter=10, \
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
    if m is None:
        m = np.ones_like(x, dtype=bool)
    if np.sum(m) < 0.1*len(m):
        return np.zeros_like(y), np.zeros(dg+1) #, m
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
        if plotting:
            plt.plot(yy)
            plt.plot(y_model)
            plt.show()
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
    return y_model, poly#, mask
