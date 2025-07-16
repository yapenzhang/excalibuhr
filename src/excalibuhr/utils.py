# File: src/excalibuhr/utils.py

import numpy as np
from astropy.io import fits
from astropy import stats
from astropy import constants as const
from astropy.modeling import models, fitting
from numpy.polynomial import polynomial as Poly
from scipy import ndimage, signal, optimize
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt 
plt.rc('image', interpolation='nearest', origin='lower')
import warnings
import os
import shutil
import subprocess
import requests


def util_master_dark(dt, combine_mode='median', badpix_clip=5):
    """
    combine dark frames; generate bad pixel map and readout noise frame
    
    Parameters
    ----------
    dt: list
        a list of dark frames to be combined 
    combine_mode: str
        the way of combining dark frames: `mean` or `median`
    badpix_clip : int
        sigma of bad pixel clipping

    Returns
    -------
    master, rons, badpix: array
        combined dark frame, readout noise frame, and bad pixel map
    """

    # Combine the darks
    if combine_mode == 'median':
        master = np.nanmedian(dt, axis=0)
    elif combine_mode == 'mean':
        master = np.nanmean(dt, axis=0)
    
    # Calculate the read-out noise as the stddev, scaled by 
    # the square-root of the number of observations
    rons = np.nanstd(dt, axis=0)/np.sqrt(len(dt))

    # Apply a sigma-clip to identify the bad pixels
    badpix = np.zeros_like(master).astype(bool)
    for i, det in enumerate(master):
        filtered_data = stats.sigma_clip(det, sigma=badpix_clip)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
    
    return master, rons, badpix


def util_master_flat(dt, dark, combine_mode='median', badpix_clip=5):
    """
    combine flat frames; generate bad pixel map
    
    Parameters
    ----------
    dt: list
        a list of flat frames to be combined 
    dark: array
        dark frame to be subtracted from the flat 
    combine_mode: str
        the way of combining flat frames: `mean` or `median`
    badpix_clip : int
        sigma of bad pixel clipping

    Returns
    -------
    master, badpix: array
        combined flat frame and bad pixel map
    """

    # Combine the flats
    if combine_mode == 'median':
        master = np.nanmedian(dt, axis=0)
    elif combine_mode == 'mean':
        master = np.nanmean(dt, axis=0)
    
    # Dark-subtract the master flat
    master -= dark

    # Apply a sigma-clip to identify the bad pixels
    badpix = np.zeros_like(master).astype(bool)
    for i, det in enumerate(master):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            filtered_data = stats.sigma_clip(det, sigma=badpix_clip)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
        # plt.imshow(badpix[i])
        # plt.show()

    return master, badpix

def combine_frames(dt, err, collapse='mean', clip=3, weights=None):
    """
    combine multiple images or spectra with error propogation 
    
    Parameters
    ----------
    dt: list
        a list of images or spectra to be combined 
    err: list
        a list of associated errors to the input data 
    collapse: str
        the way of combining frames: `mean`, `median`, `sum`. 
        `weighted` applies to spectal data, meaning a weighted average 
        by the mean SNR squared. 

    Returns
    -------
    master, master_err: array
        combined data and its error
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if collapse == 'median':
            master = np.nanmedian(dt, axis=0)
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~np.isnan(dt), axis=0)
        elif collapse == 'mean':
            # med = np.nanmedian(dt, axis=0)
            # std = np.nanstd(dt, axis=0)
            # mask = np.abs(dt - med) > clip * std
            # dt_masked = np.ma.masked_array(dt, mask=mask)
            # master = np.ma.mean(dt_masked, axis=0).data
            dt_masked = stats.sigma_clip(dt, sigma=clip, axis=0)
            master = np.ma.mean(dt_masked, axis=0).data
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~(dt_masked.mask), axis=0)
        elif collapse == 'sum':
            master = np.nansum(dt, axis=0)
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))
        elif collapse == 'weighted':
            # # weighted by average SNR squared
            dt_masked = stats.sigma_clip(dt, sigma=clip, axis=0)
            master = np.ma.average(dt_masked, axis=0, weights=weights).data
            master_err = np.sqrt(np.nansum(np.square(err), axis=0))/np.sum(~(dt_masked.mask), axis=0)


    return master, master_err


def detector_shotnoise(im, ron, GAIN=2., NDIT=1):
    """
    calculate the detector shotnoise map 
    
    Parameters
    ----------
    im: list or array
        a frame or a list of detector images     
    ron: list or array
        a frame or a list of associated readout noise maps
    GAIN: list of float or float
        detector gain for each image
    NDIT: int
        number of DIT exposure coadded

    Returns
    -------
    shotnoise: list or array
        detector shotnoise
    """
    if isinstance(GAIN, list):
        shotnoise = [np.sqrt(np.abs(im[j])/GAIN[j]/NDIT + ron[j]**2) 
                            for j in range(len(GAIN))]
    else:
        shotnoise = np.sqrt(np.abs(im)/GAIN/NDIT + ron**2)
    return shotnoise 



def flat_fielding(det, err, flat, debug=False):
    """
    apply flat fielding 
    
    Parameters
    ----------
    det: array
        detector images   
    err: array
        the assciated errors to the input images 
    flat: array
        normalized flat frame (accounting for only the pixel-to-pixel 
        variations)

    Returns
    -------
    im_corr, err_corr: array
        images and errors corrected for flat
    """
    
    im_corr = np.copy(det)
    err_corr = np.copy(err)
    badpix = np.isnan(flat).astype(bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        im_corr[~badpix] /= flat[~badpix]
        err_corr[~badpix] /= flat[~badpix]

    return im_corr, err_corr



def peak_slit_fraction(im, trace):
    """
    locate the peak signal along the slit
    
    Parameters
    ----------
    im: array
        detector image
    trace: array
        polynomials that delineate the edge of the specific order 

    Returns
    -------
    frac: float
        the fraction of peak signal along the slit 
        (frac=0 means the bottom, and 1 at the top) 
    """

    frac = []
    xx_grid = np.arange(im.shape[1])
    yy_trace = trace_polyval(xx_grid, trace)
    trace_lower, trace_upper = yy_trace
    
    for (yy_upper, yy_lower) in zip(trace_upper, trace_lower):
        # Crop out the order from the frame
        slit_len = np.mean(yy_upper-yy_lower)
        im_sub = im[int(yy_lower[len(yy_lower)//2]):
                    int(yy_upper[len(yy_lower)//2]+1)]
        # white-light psf (collpase along wavlengths)
        profile = np.nanmedian(im_sub, axis=1)
        # Pixel-location of the peak signal
        f0 = np.argmax(profile)
        frac.append(f0/slit_len)

    return np.mean(frac[1:-1])

def align_jitter(dt, err, pix_shift, tw=None, debug=False):
    """
    apply flat fielding 
    
    Parameters
    ----------
    dt: array
        detector images   
    err: array
        the assciated errors to the input images 
    pix_shift: int
        integer shift in pxiels accodring to the jitter value

    Returns
    -------
    dt_shift, err_shift: array
        shifted images and errors  
    """

    dt_shift = np.copy(dt)
    err_shift = np.copy(err)
    if pix_shift < 0:
        dt_shift[:, :pix_shift, :] = dt[:, -pix_shift:, :]
        err_shift[:, :pix_shift, :] = err[:, -pix_shift:, :]
    elif pix_shift > 0:
        dt_shift[:, pix_shift:, :] = dt[:, :-pix_shift, :]
        err_shift[:, pix_shift:, :] = err[:, :-pix_shift, :]
    if debug:
        # print peak signal location
        print(pix_shift, peak_slit_fraction(dt[0], tw[0]))

    return dt_shift, err_shift


def order_trace(det, badpix, slitlen, sub_factor=64, 
                poly_order=2, offset=0, debug=False):
    """
    Trace the spectral orders

    Parameters
    ----------
    det: array
        input flat image
    badpix: array
        bad pixel map corresponding to the `det` image
    slitlen: float
        the length of slit in pixels
    sub_factor: int
        binning factor along the dispersion axis
    offset: int
        shifting all the traces by a number of pixels.
        This is only necessary in starnge certain datasets 
        where the order edges are not monotonous.
    
    Returns
    -------
    [poly_upper, poly_lower]
        The polynomial coeffiences of upper and lower trace of each order
    """
    order_length_min = 0.75*slitlen
    width = 2

    im = np.ma.masked_array(det, mask=badpix)

    # Bin the image along the dispersion axis to reject outlier pixels
    xx = np.arange(im.shape[1])
    xx_bin = xx[::sub_factor] + (sub_factor-1)/2.
    im_clipped = stats.sigma_clip(
                            im.reshape(im.shape[0], 
                                        im.shape[1]//sub_factor, 
                                        sub_factor), 
                            axis=2)
    im_bin = np.ma.median(im_clipped, axis=2)

    # Subtract a shifted image from its un-shifted self 
    # (i.e. image gradient) to detect the trace edge
    im_grad = np.abs(im_bin[1:,:]-im_bin[:-1,:])

    # Set insignificant signal (<2sigma) to 0, only peaks are left 
    cont_std = np.nanstd(im_grad, axis=0)
    im_grad[(im_grad < cont_std*2)] = 0
    im_grad = np.nan_to_num(im_grad.data)

    xx_loc, upper, lower   = [], [], []
    # Loop over each column in the sub-sampled image
    for i in range(im_grad.shape[1]):

        # Find the peaks and separate upper and lower traces
        yy = im_grad[:,i]
        # indices = signal.argrelmax(yy)[0]
        indices, _ = signal.find_peaks(yy, distance=10) 

        # use the distance between peaks to distinguish in/between orders
        ind_distance = np.diff(indices)
        # print(ind_distance>0.9*slitlen, indices)
        upper_first = np.where(ind_distance > 0.8*slitlen)[0][-1] + 1
        ups_ind = np.arange(upper_first, 0.5, -2, dtype=int)[::-1]
        ups = indices[ups_ind]
        lows = indices[ups_ind-1]

        # check if the bottom order is complete
        # distance = ups - lows
        # if distance[0] < order_length_min:
        #     print(distance[0], order_length_min, slitlen)
        #     ups = ups[1:]
        #     lows = lows[1:]

        # Find the y-coordinates of the edges, weighted by the 
        # significance of the signal (i.e. center-of-mass)
        cens_low = np.array([np.sum(xx[int(p-width):int(p+width+1)]* \
                                    yy[int(p-width):int(p+width+1)]) / \
                            np.sum(yy[int(p-width):int(p+width+1)]) \
                            for p in lows]) + offset
        cens_up = np.array([np.sum(xx[int(p-width):int(p+width+1)]* \
                                    yy[int(p-width):int(p+width+1)]) / \
                            np.sum(yy[int(p-width):int(p+width+1)]) \
                            for p in ups]) + offset

        if debug:
            plt.plot(yy)
            for ind in cens_low:
                plt.axvline(ind, color='k')
            for ind in cens_up:
                plt.axvline(ind, color='r')
            plt.show()

        # x and y coordinates of the trace edges
        xx_loc.append(xx_bin[i])
        lower.append(cens_low)
        upper.append(cens_up)

    upper = np.array(upper).T
    lower = np.array(lower).T
    poly_upper, poly_lower = [], []
    
    # Loop over each order
    for (loc_up, loc_low) in zip(upper, lower):
        # Fit polynomials to the upper and lower edges of each order
        poly_up = Poly.polyfit(xx_loc, loc_up, poly_order)
        poly_low = Poly.polyfit(xx_loc, loc_low, poly_order)
        poly_mid = (poly_up + poly_low) / 2.

        yy_up = Poly.polyval(xx, poly_up)
        yy_low = Poly.polyval(xx, poly_low)
        yy_mid = Poly.polyval(xx, poly_mid)
        slit_len = yy_up - yy_low

        # if slit_len.min() < order_length_min:
        #     # skip the order that is incomplete
        #     continue
        if np.mean(slit_len) < 0.95*slitlen:
            # the upper or lower trace hits the edge.
            if np.mean(yy_up) > im.shape[0]*0.5:
                # print("up")
                # refine the upper trace solution with fixed slit length
                yy_up = yy_low + slitlen
                poly_up = Poly.polyfit(xx, yy_up, poly_order)
            else:
                # print("low")
                # refine the lower trace solution
                yy_low = yy_up - slit_len
                poly_low = Poly.polyfit(xx, yy_low, poly_order)
            poly_upper.append(poly_up)
            poly_lower.append(poly_low)
        else:
            # refine the trace solution by fixing the mid trace and slit length
            yy_up_new = yy_mid + slitlen / 2.
            yy_low_new = yy_mid - slitlen / 2.
            # yy_up = yy_low + slitlen
            # yy_mid_new = (yy_up + yy_low) / 2.
            # yy_up -= yy_mid_new[len(yy_mid)//2]-yy_mid[len(yy_mid)//2]
            # yy_low -= yy_mid_new[len(yy_mid)//2]-yy_mid[len(yy_mid)//2]
            poly_up = Poly.polyfit(xx, yy_up_new, poly_order)
            poly_low = Poly.polyfit(xx, yy_low_new, poly_order)
            poly_upper.append(poly_up)
            poly_lower.append(poly_low)


    print(f"-> {len(poly_upper)} orders identified")

    if debug:
        xx_grid = np.arange(0, im.shape[1])
        yy_trace = trace_polyval(xx_grid, [poly_lower, poly_upper])
        trace_lower, trace_upper = yy_trace
        plt.imshow(np.log10(im))
        for yy_lower, yy_upper in zip(trace_lower, trace_upper):
            plt.plot(xx, yy_lower, 'r')
            plt.plot(xx, yy_upper, 'r')
        plt.show()

    return [poly_lower, poly_upper]



def measure_Gaussian_center(y, peaks, width):
    """
    measure the centers of lines by fitting guassian profiles

    Parameters
    ----------
    y: array
        input spectrum   
    peaks: array
        rough locations of peaks in the input spectrum
    width: int
        window size around each peak for the profile fitting

    Returns
    -------
    center: array
        the Gaussian center of each peaks in the spectrum

    """

    xx = np.arange(len(y))
    center = np.zeros(len(peaks))
    # avoid the peaks near the edges
    peaks = peaks[(peaks<(len(xx)-width)) & (peaks>(width))]

    gg_init = models.Gaussian1D(amplitude=1, mean=0, stddev=1.) \
                + models.Const1D(amplitude=0)
    fitter = fitting.LevMarLSQFitter()

    for i in range(len(peaks)):
        y_use = y[int(peaks[i]-width):int(peaks[i]+width+1)]
        x_use = xx[int(peaks[i]-width):int(peaks[i]+width+1)]
        gg_init.mean_0 = peaks[i]
        gg_init.amplitude_0 = y[int(peaks[i])]
        gg_fit = fitter(gg_init, x_use, y_use)
        cen = gg_fit.mean_0.value
        center[i] = cen

        # plt.plot(xx, y)
        # plt.plot(xx, gg_fit(xx))
        # plt.show()
    return center

def slit_curve(fpet, une, badpix, trace, wlen_min, wlen_max, 
               sub_factor=16, une_xcorr=False, wlen_id=None, 
               debug=False):
    
    """
    Trace the curvature of the slit and determine the wavelength solution

    Parameters
    ----------
    fpet: array
        Fabry-Perot (fpet) image
    une: array
        uranium-neon lamp (une) image
    badpix: array
        bad pixel map
    trace: array
        the trace for order identification
    wlen_min, wlen_max: array
        the min and max wavelngth of each order from header for an initial 
        wavelength solution 
    sub_factor: int
        binning factor along the cross-dispersion axis
    une_xcorr: bool
        boolean flag indicating whether to use cross-correlation with the 
        une lamp to correct the wavelength solution
    wlen_id: str
        wavelength setting
    debug: bool
        boolean flag indicating the debug mode
    
    Returns
    -------
    tilt, x_fpet, wlen: array
        the slit tilt, the positions of the FP line peaks, and the corrected wavelength.
    """

    width = 35 # half width of fpet line (in pixel)
    spacing = 40 # minimum spacing (in pixel) of fpet lines
    poly_order = 2
    if wlen_id[0] == 'J':
        spacing = 30

    badpix = badpix | np.isnan(fpet)
    im = np.ma.masked_array(fpet, mask=badpix)
    im_une = np.ma.masked_array(une, mask=badpix)

    im_subs, yy_indices = im_order_cut(im, trace)

    polys_middle = np.sum(trace, axis=0)/2.

    xx = np.arange(im.shape[1], dtype=float)

    wlen, x_fpet, tilt = [], [], []
    # Loop over each order
    for (im_sub, yy, middle, w_min, w_max) in \
        zip(im_subs, yy_indices, polys_middle, wlen_min, wlen_max):

        im_sub[im_sub.mask] = 0.

        slit_image, x_slit, poly_slit = [], [], []
        N_lines = 0
        # Loop over each pixel-row in a sub-sampled image
        for row in range(yy[0], yy[-1]-sub_factor//2, sub_factor):

            # combine a few rows (N=sub_factor) to increase 
            # S/N and reject outliers
            im_masked = stats.sigma_clip(im[row:row+sub_factor], 
                                sigma=2, axis=0, masked=True)
            spec_fpet = np.ma.mean(im_masked, axis=0)

            # find bad channels where 70% of rows are masked
            mask = (np.sum(im_masked.mask, axis=0) > 0.7*sub_factor)
            badchannel = np.argwhere(mask)[:,0]
            # mask neighboring pixels as well
            badchannel = np.concatenate((badchannel, 
                            badchannel+1, badchannel-1), axis=None)

            # measure the baseline of the spec to be 10% lowest values
            height = np.median(np.sort(spec_fpet.data[~mask])[:len(spec_fpet)//10])

            # Find the peak pixels (along horizontal axis) the width can be a problem
            peaks, properties = signal.find_peaks(spec_fpet, distance=spacing, 
                                        width=5, height=2*height) 
            # mask the peaks identified due to bad channels
            peaks = np.array([item for item in peaks if not item in badchannel])

            # leave out lines around detector edge
            peaks = peaks[(peaks<(im.shape[1]-width)) & (peaks>(width))]

            # plt.plot(spec_fpet)
            # for p in peaks:
            #     plt.axvline(p, color='r')
            # plt.show()

            # Calculate center-of-mass of the peaks
            width = np.median(properties['widths'])
            cens = measure_Gaussian_center(spec_fpet, peaks, width=width)

            slit_image.extend([[p, row+(sub_factor-1)/2.] for p in cens])

            # generate bins to divide the peaks in groups
            # when maximum number of fpet lines are identified in the order
            if len(peaks) > N_lines:
                bins = sorted([x-spacing for x in cens] + \
                              [x+spacing for x in cens])
                N_lines = len(peaks)
            
        slit_image = np.array(slit_image)
        # Index of bin to which each peak belongs
        indices = np.digitize(slit_image[:,0], bins)

        # Loop over every other bin
        for i in range(1, len(bins), 2):
            xs = slit_image[:,0][indices == i] # x-coordinate of peaks
            ys = slit_image[:,1][indices == i] # y-coordinate 
            
            if len(xs) > poly_order:
                # plt.scatter(xs, ys)

                # Fit a polynomial to the the fpet signal
                poly = Poly.polyfit(ys, xs, poly_order)
                poly_orth = Poly.polyfit(xs, ys, poly_order)

                # Find mid-point on slit image, i.e. 
                # the intersection of two polynomials
                root = Poly.polyroots(np.pad(poly_orth, 
                        (0, len(middle)-len(poly_orth)), 'constant')
                        - middle)

                if np.iscomplexobj(root):
                    # slit is close to vertical 
                    root = np.copy(xs)
                # Select the intersection within the valid x-coordinates
                root = root[(root>int(xs.min()-2)) & (root<int(xs.max()+2))]

                # x_slit stores the centre x-coordinates of each fpet line 
                if len(root)>0:
                    x_slit.append(root.mean())
                    poly_slit.append(poly)

        # Account for the variation of the polynomial coefficients along wavelengths
        poly_slit = np.array(poly_slit)
        tilt.append([Poly.polyfit(x_slit, poly_slit[:,i], poly_order) 
                        for i in range(poly_order+1)])
        x_fpet.append(x_slit)

        # plt.imshow(im, vmin=0, vmax=5e3)
        # plt.show()

        # Determine the wavelength solution along the detectors/orders
        # Mapping from measured fpet positions to a linear-spaced grid, 
        # fitting with 2nd-order poly
        ii = np.arange(len(x_slit))
        poly = Poly.polyfit(x_slit, ii, 2)
        
        # # Fix the end of the wavelength grid to values from the header
        # grid_poly = Poly.polyfit([Poly.polyval(xx, poly)[0], 
        #                           Poly.polyval(xx, poly)[-1]], 
        #                          [w_min, w_max], 1)
        # ww_cal = Poly.polyval(Poly.polyval(xx, poly), grid_poly)
        # wlen.append(ww_cal)

        # Fix the end of the frequency grid to values based on the header
        # from Kevin Hoy
        nmc = const.c.to("nm/s").value
        f_min = nmc / w_min
        f_max = nmc / w_max
        grid_poly = Poly.polyfit([Poly.polyval(xx, poly)[0],
                                Poly.polyval(xx, poly)[-1]],
                                [f_min, f_max], 1)
        ff_cal = Poly.polyval(Poly.polyval(xx, poly), grid_poly)
        ww_cal = nmc / ff_cal
        wlen.append(ww_cal)

        # if debug:
        #     plt.plot(np.diff(x_slit))
        #     plt.show()
        #     freq_slit = Poly.polyval(Poly.polyval(x_slit, poly), grid_poly)
        #     plt.plot(np.diff(freq_slit)/freq_slit[0])
        #     plt.show()

    # check rectified image
    # spectral_rectify_interp(im, badpix, trace, tilt, debug=True)

    if une_xcorr:
        # extract une spectrum
        wlen_xcorr = []
        x_model, y_model, y_model_s, x_lines, w0s, w1s = genline(wlen, wlen_id)
        template_interp_func = interp1d(x_model, y_model_s, kind='linear', 
                                        bounds_error=False, fill_value=0)
        spec_une, _ = extract_blaze(im_une, badpix, trace)
        dw = 0.06
        order = 2
        p_range=[0.1, 0.01, 0.01]
        bound = [(-p_range[j], p_range[j]) for j in range(order+1)]

        for (flux, w_init) in zip(spec_une, wlen):
            f = np.zeros_like(w_init)
            order_mask = (w0s > w_init[0]) & (w1s < w_init[-1])
            if not np.any(order_mask):
                w_cal = w_init
            else:
                # estimate the continuum level from the neighbourhood of each line
                for w0, w1 in zip(w0s[order_mask], w1s[order_mask]):
                    cont_mask = ((w_init > w0 - dw) & (w_init < w0)) \
                            | ((w_init < w1 + dw) & (w_init > w1))
                    line_mask = (w_init > w0) & (w_init < w1)
                    poly_cont = Poly.polyfit(w_init[cont_mask], flux[cont_mask], 1)
                    flux[line_mask] -= Poly.polyval(w_init[line_mask], poly_cont)
                    # plt.plot(w_init[cont_mask], flux[cont_mask],'r', zorder=10)

                # mark the regions of lines for cross-correlation with lamp model
                line_mask = np.zeros_like(w_init)
                order_mask = (x_lines > w_init[0])&(x_lines<w_init[-1])
                for l in x_lines[order_mask]:
                    line_mask += (w_init > l - dw) & (w_init < l + dw)
                line_mask = line_mask.astype(bool)

                f[line_mask] = flux[line_mask]

                # find optimal wavelength solution correction 
                # by maximizing the cross-correlation
                res = optimize.minimize(
                                func_wlen_optimization, 
                                args=(w_init, f, template_interp_func), 
                                x0=np.zeros(order+1), method='Nelder-Mead', 
                                bounds=bound) 
                poly_opt = res.x
                # print(poly_opt)
                w_cal = w_init + \
                            Poly.polyval(w_init - np.mean(w_init), poly_opt) 

            wlen_xcorr.append(w_cal)

            # plt.plot(w_init, flux)
            # plt.plot(w_init, f)
            # plt.plot(w_cal, flux)
            # plt.plot(w_cal, template_interp_func(w_cal), 'k', alpha=0.5)
            # plt.show()
        wlen = wlen_xcorr

    return tilt, x_fpet, wlen



def genline(wlen, wlen_id):
    """
    generate lamp model spectra for specific wavelength settings from linelist

    Parameters
    ----------
    wlen: array
        an initial guess of the observed wavelength
    wlen_id: str
        wavelength setting identifier  

    Returns
    -------
    x_model, y_model: array
        model spectrum generated from the lamp linelist
    wave: array
        center of selected lamp lines
    w0s, w1s: array
        minimum and maximum wavelengths of selected regions with lamp lines

    """

    def G(x, x0, a, sigma):
        """ Return Gaussian line shape wth sigma """
        # return a / np.sqrt(2. * np.pi) / sigma\
        return a * np.exp(-( (x - x0) / sigma)**2 / 2.)

    wlen = np.array(wlen)
    w_min = wlen[:,0]
    w_max = wlen[:,-1]

    # read the calibration lamp data
    src_path = os.path.dirname(os.path.abspath(__file__))
    if np.max(w_max) < 1440:
        lines_file = os.path.join(src_path, '../../data/lines_u_sarmiento.txt')
    else:
        lines_file = os.path.join(src_path, '../../data/lines_u_redman.txt')

    selection_file =  os.path.join(src_path, f'../../data/{wlen_id}.dat')
    w0s, w1s = np.genfromtxt(selection_file, unpack=1)

    wave, amp, = [], []
    with open(lines_file, 'r') as fp:
        dt = fp.readlines()
        for x in dt:
            wave.append(float(x[:9]))
            amp.append(float(x[9:]))
    wave, amp = np.array(wave), np.array(amp)

    # select wavelength regions based on the observations
    indices = []
    for w0, w1 in zip(w_min, w_max):
        indices.append((wave>w0) & (wave<w1))
    indices = np.sum(indices, axis=0).astype(bool)
    wave, amp = wave[indices], amp[indices]

    # make the lamp model spectrum from the line list
    x_model = np.arange(wave[0]-10, wave[-1]+10, 0.002)
    y_model = np.zeros_like(x_model)
    for i in range(len(wave)):
        y_model += G(x_model, wave[i], 10., 0.01)
    
    indices = []
    for w0, w1 in zip(w0s, w1s):
        indices.append((wave>w0) & (wave<w1))
    indices = np.sum(indices, axis=0).astype(bool)
    wave, amp = wave[indices], amp[indices]

    # make the lamp model spectrum from the line list
    y_model_s = np.zeros_like(x_model)
    for i in range(len(wave)):
        # if amp[i]>10:
            y_model_s += G(x_model, wave[i], 10., 0.01)

    return x_model, y_model, y_model_s, wave, w0s, w1s



def trace_polyval(xx, tw):
    """
    evaluate trace polynomials given the pixel grid

    Parameters
    ----------
    xx: array
        pixel grid
    tw: array
        polynomial coefficiences for the upper and lower traces 
        for mutiple orders on one detector image  

    Returns
    -------
    yy_list: list
        the evaluated y coordinates of upper and lower order edges
    """

    yy_list = []
    for trace in tw:
        yy_order = []
        for poly in trace:
            yy = Poly.polyval(xx, poly)
            if np.any(yy > len(xx)):
                yy = np.zeros_like(yy) + len(xx) - 4.
            elif np.any(yy < 0):
                yy = np.zeros_like(yy) + 4.
            yy_order.append(yy)
        yy_list.append(yy_order)
    return yy_list


                      


def im_order_cut(im, trace):
    """
    Cut out individual orders from the detector image

    Parameters
    ----------
    im: array
        input image
    trace: array
        polynomials that delineate the edge of the specific order 

    Returns
    -------
    im_subs: array
        sub-images of each order
    yy_indices: array
        y-axis indices of each order
    """

    # Evaluate order traces 
    xx = np.arange(im.shape[-1])
    yy_trace = trace_polyval(xx, trace)
    trace_lower, trace_upper = yy_trace

    yy_indices, im_subs, xx_shifts = [], [], []
    for o, (yy_upper, yy_lower) in enumerate(zip(trace_upper, trace_lower)):
        im_sub = im[int(yy_lower.min()):int(yy_upper.max()+1)]
        yy_indice = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))
        im_subs.append(im_sub)
        yy_indices.append(yy_indice)

    return im_subs, yy_indices



def slit_polyval(xx, meta):
    """
    evaluate slit curvature polynomials given the pixel grid

    Parameters
    ----------
    xx: array or list
        pixel grid or a list of x coordinates
    meta: array
        meta polynomial coefficiences for the change of slit tilt over 
        x coordinates

    Returns
    -------
    slit: list
        the evaluated polynomial coefficiences describing the slit tilt 
        for each given x coordinate
    """

    if isinstance(xx, list):
        slit = [np.array([Poly.polyval(x_line, poly_meta) 
                          for poly_meta in meta_order]).T 
                for (x_line, meta_order) in zip(xx, meta)]
    else:
        slit = [np.array([Poly.polyval(xx, poly_meta) 
                          for poly_meta in meta_order]).T 
                for meta_order in meta]
    return slit


def spectral_rectify_interp(im_list, badpix, trace, slit_meta, reverse=False, debug=False):
    """
    Correct for the slit-tilt by interpolating to a pixel-grid

    Parameters
    ----------
    im_list: array
        input image or a list of image and variance to be corrected
    badpix: array
        bad pixel map corresponding to the image
    trace: array
        polynomials that delineate the edge of the specific order 
    slit_meta: array
        polynomials that describing the slit curvature 
        as a function of the dispersion axis
    reverse: bool
        if `True`, interpolate back to the tilted slit 

    Returns
    -------
    im_rect_spec: array
        rectified images where the slit image is vertical on detector
    """

    if np.array(im_list).ndim == 3:
        im_rect_spec = np.array(im_list)
    elif np.array(im_list).ndim == 2:
        im_rect_spec = np.array(im_list)[np.newaxis, :]
    else:
        raise TypeError("Invalid data dimension")

    bpm = np.logical_or(badpix, np.isnan(im_rect_spec[0]))

    # Evaluate order traces and slit curvature
    xx_grid = np.arange(0, im_rect_spec.shape[-1])
    _, yy_indices = im_order_cut(im_rect_spec[0], trace)
    slit_poly = slit_polyval(xx_grid, slit_meta)

    # Loop over each order
    for (yy_grid, poly_full) in zip(yy_indices, slit_poly):

        # only focus on the positive part of the signal 
        # doesn't work in the case of wide companions
        # profile = np.nanmedian(im_rect_spec[0, yy_grid], axis=1)
        # if np.argmax(profile) - len(yy_grid)//2 < 0:
        #     yy_half = yy_grid[:len(yy_grid)//3*2]
        #     profile = profile[:len(yy_grid)//3*2]
        # else:
        #     yy_half = yy_grid[len(yy_grid)//3*2:]
        #     profile = profile[len(yy_grid)//3*2:]

        # create the grid for the tilted slit 
        isowlen_grid = np.zeros((len(yy_grid), len(xx_grid)))
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])

        # Loop over each row in the order
        for i, (x_isowlen, mask) in enumerate(zip(isowlen_grid, bpm[yy_grid])):
            
            if np.sum(mask)>0.5*len(mask):
                continue

            data_row = im_rect_spec[:, yy_grid[i]]

            # Correct for the slit-curvature by interpolating onto the grid
            # loop over each image
            for r, dt in enumerate(data_row):
                if reverse:
                    im_rect_spec[r, yy_grid[i]] = interp1d(x_isowlen[~mask], 
                                                        dt[~mask], 
                                                        kind='linear', 
                                                        bounds_error=False, 
                                                        fill_value=np.nan
                                                        )(xx_grid)
                else:
                    im_rect_spec[r, yy_grid[i]] = interp1d(xx_grid[~mask], 
                                                        dt[~mask], 
                                                        kind='linear', 
                                                        bounds_error=False, 
                                                        fill_value=np.nan
                                                        )(x_isowlen)
                                                       
    if debug:
        plt.imshow(im_rect_spec[0], vmin=0, vmax=1e3)
        plt.show()
    
    if np.array(im_list).ndim == 2:
        return im_rect_spec[0]
    else:
        return im_rect_spec



def trace_rectify_interp(im_list, trace, debug=False):
    """
    Correct for the curvature of traces by interpolating to a pixel-grid
    pivoting on the middle of the detector

    Parameters
    ----------
    im_list: array
        input image or a list of images to be corrected
    trace: array
        polynomials that delineate the edge of the specific order 

    Returns
    -------
    im_rect: array
        rectified images where the trace is horizontal on detector
    """

    if np.array(im_list).ndim == 3:
        im_rect = np.array(im_list)
    elif np.array(im_list).ndim == 2:
        im_rect = np.array(im_list)[np.newaxis, :]
    else:
        raise TypeError("Invalid data dimension")

    xx_grid = np.arange(0, im_rect.shape[-1])
    yy_trace = trace_polyval(xx_grid, trace)
    trace_lower, trace_upper = yy_trace
    trace_mid = trace_polyval(xx_grid, np.mean(trace, axis=0)[np.newaxis,:])[0]

    for (yy_upper, yy_lower, yy_mid) in zip(trace_upper, trace_lower, trace_mid):
        shifts = yy_mid - yy_mid[len(xx_grid)//2] 
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        for x in xx_grid:
            data_col = im_rect[:,yy_grid,x]
            for r, dt in enumerate(data_col):
                mask = np.isnan(dt)
                if np.sum(mask)>0.5*len(mask):
                    continue 
                im_rect[r,yy_grid,x] = interp1d(yy_grid[~mask], dt[~mask], 
                                                kind='linear', 
                                                bounds_error=False, 
                                                fill_value=np.nan
                                                )(yy_grid+shifts[x])

    return im_rect


def master_flat_norm(det, badpix, trace, slit_meta, slitlen=None, debug=False):
    """
    normalize master flat frame and extract balze function
    
    Parameters
    ----------
    det: array
        master flat frame 
    badpix: array
        bad pixel map corresponding to the input image
    trace: array
        polynomials that delineate the edge of the specific order 
    slit_meta: array
        polynomials that describing the slit curvature as a 
        function of the dispersion axis
    slitlen: float
        the length of slit in pixels

    Returns
    -------
    flat_norm, blazes: array
        normalized flat frame and blaze functions
    """

    # # Correct for the slit curvature
    # det_rect = spectral_rectify_interp(det, badpix, trace, slit_meta)
    # filtered_data = stats.sigma_clip(det_rect, sigma=5)
    # badpix_rect = filtered_data.mask | badpix
    # # measure the trace again
    # trace_update = order_trace(det_rect, badpix_rect, slitlen=slitlen)

    # Retrieve the blaze function by mean-collapsing 
    # the master flat along the slit
    blaze, blaze_image = extract_blaze(det, badpix, trace)
    # blaze, blaze_image = extract_blaze(det_rect, badpix_rect, trace_update)

    # blaze_image = spectral_rectify_interp(blaze_image, np.zeros_like(badpix), 
    #                     trace_update, slit_meta, reverse=True)
    # plt.imshow(blaze_image, vmin=1e1, vmax=2e4)
    # plt.show()
    # Set low signal to NaN
    flat_norm = det / blaze_image
    # flat_norm[flat_norm<0.9] = np.nan
    # flat_norm[flat_norm>1.1] = np.nan

    # if debug:
        # plt.imshow(flat_norm, vmin=0.8, vmax=1.2)
        # plt.show()

    return flat_norm, blaze, trace

def extract_blaze(im, badpix, trace, f0=0.5, fw=0.48, sigma=3):
    """
    Collpase 2D image along the cross-dispersion direction 
    in an order-by-order basis.

    Parameters
    ----------
    im: array
        input flat image to be normalized
    badpix: array
        bad pixel map corresponding to the input image
    trace: array
        polynomials that delineate the edge of the specific order 

    Returns
    -------
    blaze_orders: array
        1D blaze function of each order
    """
    badpix = badpix.astype(bool)

    xx_grid = np.arange(0, im.shape[1])
    yy_trace = trace_polyval(xx_grid, trace)
    trace_lower, trace_upper = yy_trace

    blaze_image = np.ones_like(im)*np.nan
    blaze_orders = []
    # Loop over each order
    for (yy_upper, yy_lower) in zip(trace_upper, trace_lower):

        blaze = np.zeros_like(xx_grid, dtype=np.float64)
        indices = [range(int(low+(f0-fw)*(up-low)), 
                             int(low+(f0+fw)*(up-low))+1) \
                        for (up, low) in zip(yy_upper, yy_lower)]

        # Loop over each column
        for i, ind in enumerate(indices):
            mask = badpix[ind,i] | np.isnan(im[ind,i])
            if np.sum(mask)>0.5*len(mask):
                blaze[i] = np.nan
            else:
                # get the cross-dispersion mean value 
                blaze[i], _, _ = stats.sigma_clipped_stats(
                    im[ind,i][~mask], sigma=sigma)
        
        # smooth blaze
        nans = np.isnan(blaze)
        smoothed , _, _ = PolyfitClip(xx_grid, blaze, order=13, mask=~nans)
        # plt.plot(xx_grid, blaze/np.nanmax(blaze))
        # plt.plot(xx_grid, smoothed/np.nanmax(blaze))
        # plt.show()
            
        for i, ind in enumerate(indices):
            # put blaze back to 2D image
            blaze_image[ind,i] = smoothed[i]
        blaze_orders.append(smoothed)

    return blaze_orders, blaze_image



def readout_artifact(det, det_err, badpix, trace, Nborder=20, sigma=3, debug=False):
    """
    Correct for readout noise artifacts that appear like vertical strips 
    by subtracting the inter-order average value in each column on detector.

    Parameters
    ----------
    det: array
        input science image
    det_err: array
        errors associated with the input science image
    badpix: array
        bad pixel map corresponding to the `det` image
    trace: array
        polynomials that delineate the edge of the specific order 
    Nborder: int
        number of pixels to skip at the border of orders

    Returns
    -------
    det, det_err: array
        det and det_err corrected for the detector artifact
    """

    badpix = (badpix | np.isnan(det))
    det[badpix] = np.nan
    det_err[badpix] = np.nan

    xx = np.arange(det.shape[1])
    yy_trace = trace_polyval(xx, trace)
    trace_lower, trace_upper = yy_trace
    lowers = [np.min(item) for item in trace_lower]
    uppers = [np.max(item) for item in trace_upper]
    uppers = uppers[:-1]
    lowers = lowers[1:]
    
    indices_row = []
    for up, low in zip(uppers, lowers):
        indices_row += list(range(int(up+Nborder), int(low-Nborder+1)))

    # # use rows without light
    # indices_row = range(5, 40)

    im = stats.sigma_clip(det[indices_row], sigma=sigma, axis=0)
    im_err = np.ma.masked_array(det_err[indices_row], mask=im.mask)
    ron_col = np.ma.mean(im, axis=0)
    err_col = np.sqrt(np.ma.sum(im_err**2, axis=0)) / np.sum(~im.mask, axis=0)
    
    det -= ron_col
    det_err = np.sqrt(det_err**2+err_col**2)

    if debug:
        plt.imshow(det, vmin=-20, vmax=20)
        plt.show()
    return det, det_err


def remove_skylight(D_full, V_full, M_bpm, obj_cen, frac_mask, debug=False):
    """
    Remove sky background in a 2D detector image. 

    Parameters
    ----------
    D_full: array
        cropped image of one order 
    V_full: array
        Variance of the input image accounting for sky background 
        and readout noise
    M_bpm: array
        bad pixel map corresponding to the detector image
    obj_cen: float
        location of the star on slit in pixel 
    frac_mask: float
        half width of the trace to be masked in fraction of slit length

    Returns
    -------
    D_cor, err_cor: array
        image and error after the sky background removal

    """

    # mask trace of the target before combining the background
    yy = np.arange(D_full.shape[0])
    width = int(frac_mask*len(yy))
    D = np.ma.masked_array(D_full, mask=M_bpm)
    D.mask[max(obj_cen-width, 0):obj_cen+width+1, :] = True 
    Nedge = 5
    D.mask[:Nedge,:] = True
    D.mask[-Nedge:,:] = True
    bkg_image = stats.sigma_clip(D, axis=0)
    bkg = np.ma.mean(bkg_image, axis=0)
    # plt.imshow(bkg_image, aspect='auto', vmin=0, vmax=100)
    # plt.show()

    # err propogation
    V = np.ma.masked_array(V_full, mask=bkg_image.mask)
    bkg_var = np.ma.sum(V, axis=0)/np.sum(~V.mask, axis=0)**2

    D_cor = D_full - bkg.data
    V_cor = V_full + bkg_var.data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        err_cor = np.sqrt(V_cor)

    if debug:
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.imshow(D_full, aspect='auto', vmin=-10, vmax=100)
        ax2.imshow(D_cor, aspect='auto', vmin=-10, vmax=100)
        plt.show()
    return D_cor, err_cor


def remove_starlight(D_full, V_full, spec_star, cen0_p, cen1_p, 
                    aper0=40, aper1=10, sub_factor=32, 
                    gain=2.0, NDIT=1., debug=False):
    """
    [EXPERIMENTAL!] Remove starlight contamination at the position of 
    the widely separated companion in a 2D detector image. 

    Parameters
    ----------
    D_full: array
        cropped image of one order 
    V_full: array
        Variance of the input image accounting for sky background 
        and readout noise
    spec_star: array
        in case of sub-stellar companion extraction and `remove_star_bkg=True`,
        provide the extracted stellar spectra. This will be scaled according  
        to the PSF and then removed from the science image before extracting
        the companion spectra.
    cen0_p: float
        location of the star on slit in pixel 
    cen1_p: float
        location of the companion on slit in pixel
    aper0: int
        half aperture for masking the negative primary trace
    aper1: int
        half aperture for masking the negative secondary trace 
        and the primary line core
    aper2: int
        half aperture for determining the line center
    sub_factor: int
        binning factor along the dispersion axis
    gain: float
        detector gain
    NDIT: int
        number of DIT exposure coadded

    Returns
    -------
    D_full, Err_full: array
        image and error after the starlight removal
    """
    # determine the location of peak signal from data
    D_full = np.nan_to_num(D_full)
    spatial_x = np.arange(len(D_full))
    bkg = np.zeros_like(D_full)

    if cen0_p + aper0 > len(spatial_x):
        return D_full, np.sqrt(V_full)


    polys = []
    x_sub = np.arange(0, D_full.shape[-1], sub_factor) + sub_factor/2.
    D = np.reshape(D_full, (D_full.shape[0], 
                            D_full.shape[-1]//sub_factor, 
                            sub_factor))
    D_sub = np.nanmedian(D, axis=2)

    # measure the center of the PSF peak
    cen0_n = np.argmax(np.nanmedian(-D_sub, axis=1))
    cen1_n = int(cen0_n - cen0_p + cen1_p)
    profile = np.nanmedian(D_sub, axis=1)
    cen0_p = measure_Gaussian_center(profile, np.array([cen0_p]), width=5)[0]

    # aper0: mask the negative primary trace
    # aper1: mask the negative secondary trace and the primary line core
    mask =  (((spatial_x>cen1_n-aper1) & (spatial_x<cen1_n+aper1)) | \
            ((spatial_x>cen0_n-aper0) & (spatial_x<cen0_n+aper0))) | \
            ((spatial_x>cen0_p-aper1) & (spatial_x<cen0_p+aper1)) 
    
    # check at which side the secondary is located
    if cen1_p - cen0_p > 0: 
        m = (~mask) & (spatial_x-cen0_p<0)
    else:
        m = (~mask) & (spatial_x-cen0_p>0)
    
    # flip PSF to the oposite side
    xx = 2.*cen0_p - spatial_x[m]
    D_sub = D_sub[m]

    if debug:
        # print(cen0_p, cen1_p, cen0_n, cen1_n)
        plt.plot(spatial_x, profile)
        # plt.plot(spatial_x[mask], profile[mask])
        plt.plot(spatial_x[m], profile[m])
        plt.plot(xx, profile[m])
        plt.show()

    for w in range(len(x_sub)):
        poly = Poly.polyfit(xx, D_sub[:,w], 5)
        # plt.plot(xx, D_sub[:,w])
        # plt.plot(xx, Poly.polyval(xx, poly))
        # plt.show()
        polys.append(poly)

    polys = np.array(polys)
    polys_model = np.zeros((D_full.shape[-1], polys.shape[-1]))
    # suppose each polynomial coeffecient vary with wavelength linearly
    # evaluate polynomials at the full x coordinates (dispersion axis)
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

    bkg = bkg * np.nan_to_num(spec_star) / np.nanmedian(spec_star)

    if debug:
        plt.plot(profile)
        plt.plot(profile_model)
        plt.show()
        plt.imshow(D_full, vmin=-10, vmax=20, aspect='auto')
        plt.show()
        plt.imshow(D_full-bkg, vmin=-10, vmax=20, aspect='auto')
        plt.show()

    return D_full-bkg, Err_full



def extract_spec(det, det_err, badpix, trace, slit, blaze, gain, NDIT=1, 
                    cen0=90, 
                    companion_sep=None, 
                    aper_half=20, 
                    extract_2d=False, 
                    extr_level=0.9,
                    remove_star_bkg=False, 
                    remove_sky_bkg=False, 
                    debug=False):
    """
    Extract 1D spectra from detector images

    Parameters
    ----------
    im: array
        input science image that has been calibrated
    im_err: array
        readout+background noise associated with the input science image
    badpix: array
        bad pixel map corresponding to the input image
    trace: array
        polynomials that delineate the edge of the specific order 
    slit: array
        polynomials that describing the slit curvature 
        as a function of the dispersion axis
    blaze: array
        1D blaze function of each order
    gain: float
        detector gain
    NDIT: int
        number of DIT exposure coadded
    cen0: float
        location of the star on slit in pixel 
    companion_sep: float
        angular separation of the companion on slit in pixel
    aper_half: int
        half of extraction aperture in pixel
    extract_2d: bool
        if extracting 2D spectra, the curvature of the trace will be explicitly corrected
    interpolation: bool
        This determines the mode of the slit curvature correction.
        If `False`, the 2D spectra will be resampled to a normal grid to correct for the slit curvature.
        Otherwise, it uses interpolation to speed up.
    remove_star_bkg: bool
        in case of sub-stellar companion extraction, whether remove the starlight contamination.
    remove_sky_bkg: bool
        in case of staring observations, set it to `True` to remove the sky background.

    Returns
    -------
    flux, err: array
        the extracted fluxes and their uncertainties
    D_stack, P_stack, V_stack: array
        2d spectral data, modeled 2d slit function, 2d spectral uncertainty
    id_order: array
        the output 2D data of different orders are stacked along the first axis, 
        `id_order` contains the indices to retrieve the data of each order.
    chi2: array
        reduced chi2 of the model (for diagnositic purposes)
    """
    # Correct for the slit curvature and trace curvature
    det_rect = spectral_rectify_interp([det, det_err], badpix, trace, slit, debug=False)
    im, im_err = det_rect

    # plt.imshow(im_err, vmin=0, vmax=3e1, aspect='auto')
    # plt.show()
    #TODO how to deal with interpolation of badpix

    if extract_2d or remove_star_bkg or remove_sky_bkg:
        dt_rect = trace_rectify_interp([im, im_err], trace, debug=False) 
        im, im_err = dt_rect
        filter_mode = 'median' # rectifying the trace using interpolation will 
        # introduce low frequency trend in the dispersion direction, 
        # which need to be taken into account when modeling the psf
    else:
        filter_mode = 'poly'

    # cut images to orders
    im_subs, yy_indices = im_order_cut(im, trace)
    im_err_subs, yy_indices = im_order_cut(im_err, trace)
    bpm_subs, yy_indices = im_order_cut(badpix, trace)
    
    flux, err, D, P, V = [],[],[],[],[]
    for o, (im_sub, im_err_sub, bpm_sub) in enumerate(zip(im_subs, im_err_subs, bpm_subs)):
        
        obj_cen = cen0
        # find out the location of the peak signal in each order
        profile = np.nanmedian(im_sub, axis=1)
        # avoid edges
        N_edge = 10
        profile[:N_edge] = 0.
        profile[-N_edge:] = 0.
        if cen0 is None:
            obj_cen = np.argmax(profile)
        else:
            # if the extraction location is given, then adjust slightly around it
            profile[:obj_cen-4] = -np.inf
            profile[obj_cen+4:] = -np.inf
            obj_cen = np.argmax(profile) 


        if remove_sky_bkg:
            im_sub, im_err_sub = remove_skylight(
                                im_sub, im_err_sub**2, bpm_sub,
                                obj_cen=obj_cen, 
                                frac_mask=0.35,
                                debug=debug)
        
        # remove starlight contamination
        # if remove_star_bkg and companion_sep is not None:
        #     im_sub, im_err_sub = remove_starlight(im_sub, im_err_sub**2, 
        #                     spec_star[o]/blaze[o], cen0, cen0-companion_sep, 
        #                     gain=gain, debug=debug)
        #     aper_half = 5

        # Pixel-location of the companion target
        if companion_sep is not None:
            obj_cen -= companion_sep

        # Extract a 1D spectrum using the optimal extraction algorithm
        f_opt, f_err, D_sub, V_sub, P_sub = optimal_extraction(
                                im_sub.T, im_err_sub.T**2, bpm_sub.T, 
                                obj_cen=int(np.round(obj_cen)), 
                                aper_half=aper_half, 
                                filter_mode=filter_mode,
                                remove_bkg=remove_star_bkg,
                                extr_level=extr_level,
                                gain=gain, NDIT=NDIT, debug=debug) 

        flux.append(f_opt/blaze[o])
        err.append(f_err/blaze[o])
        D.append(D_sub)
        V.append(V_sub)
        P.append(P_sub)

    return flux, err, D, V, P 

        
def optimal_extraction(D_full, V_full, bpm_full, obj_cen, 
                       aper_half=20, filter_mode='poly',
                       badpix_clip=5, filter_width=121,
                       max_iter=30, extr_level=0.9, 
                       remove_bkg=False, etol=1e-20, 
                       gain=2., NDIT=1., debug=False):
    """
    Optimal extraction based on Horne(1986).

    Parameters
    ----------
    D_full: array
        cropped image of one order 
    V_full: array
        Variance of the input image accounting for background 
        and readout noise
    bpm_full: array
        bad pixel map corresponding to the input image
    obj_cen: float
        location of the target on slit in fraction between [0,1] 
    aper_half: int
        half of extraction aperture in pixels
    badpix_clip: int
        sigma of bad pixel clipping 
    max_iter: int
        maximum number of iterations for bad pixel or cosmic ray rejection
    gain: float
        detector gain
    NDIT: int
        number of DIT exposure coadded
    etol: float
        The tolerance parameter to avoid division by 0
    
    Returns
    -------
    f_opt, f_err: array
        the extracted fluxes and their uncertainties
    D, chi2_r: array
        modeled slit function and reduced chi2 of the model (for plotting)
    """

    D = D_full[:,obj_cen-aper_half:obj_cen+aper_half+1] # Observation
    V = V_full[:,obj_cen-aper_half:obj_cen+aper_half+1] # Variance
    bpm = bpm_full[:,obj_cen-aper_half:obj_cen+aper_half+1]

    if D.size == 0:
        # print("Trace falls outside of the detector")
        return np.zeros(D.shape[0]), np.zeros(D.shape[0]), \
                np.zeros(D.T.shape), np.zeros(D.T.shape), \
                np.zeros(D.T.shape)

    D = np.nan_to_num(D, nan=etol)
    V = np.nan_to_num(V, nan=1./etol)
    V_new = V + np.abs(D) / gain / NDIT
    
    # plt.imshow(V, aspect='auto', vmin=0, vmax=20)
    # plt.show()

    wave_x = np.arange(D.shape[0])
    spatial_x = np.arange(D.shape[1])
    D_norm = np.zeros_like(D)

    # fit and remove bkg per wavelength channel
    if remove_bkg == True:
        
        # may need to tweek this parameter
        # 5th order polynomial gives the best performance so far
        bkg_poly_order = 6 
        aper_mask = 6 
        cen0 = len(spatial_x)//2
        mask = (spatial_x > cen0 - aper_mask) & (spatial_x < cen0 + aper_mask)
        bkg_model = np.zeros_like(D)

        filtered_data = stats.sigma_clip(D[:, ~mask], sigma=3, axis=1)
        bpm[:,~mask] = np.logical_or(bpm[:,~mask], filtered_data.mask)

        for x in wave_x:
            m = np.logical_or(mask, bpm[x])
            y_model, _, _ = PolyfitClip(spatial_x, D[x], order=bkg_poly_order, mask=(~m))
            bkg_model[x] = y_model
            # plt.plot(spatial_x, D[x])
            # plt.plot(spatial_x[~m], D[x, ~m])
            # plt.plot(spatial_x, y_model)
            # # plt.plot(spatial_x[~mask], D_poly)
            # plt.show()
        if debug:
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharex=ax1)
            ax3 = plt.subplot(313, sharex=ax1)
            ax1.imshow(D.T, aspect='auto', vmin=0, vmax=20)
            ax2.imshow(bkg_model.T, aspect='auto', vmin=0, vmax=20)
            ax3.imshow(D.T-bkg_model.T, aspect='auto', vmin=0, vmax=20)
            plt.show()

        D -= bkg_model
        V_new += np.abs(bkg_model) / gain / NDIT


    # simple sum collapse to a 1D spectrum (box extraction)
    f_std = np.nansum(D*np.logical_not(bpm).astype(float), axis=1)

    # Normalize the image per spatial row with the simple 1D spectrum
    # for the estimation of the spatial profile P
    for x in spatial_x:
        D_norm[:,x] = D[:,x]/(f_std+etol)

    P = np.zeros_like(D)
    # Fit each row with a polynomial or median filter while clipping bad pixels
    for x in spatial_x:
        y = D_norm[:,x]
        if filter_mode == 'poly':
            P[:,x], _, _ = PolyfitClip(wave_x, y, 12)
            # plt.plot(wave_x, y)
            # plt.plot(wave_x, P[:,x])
            # plt.ylim(0, 0.2)
            # plt.show()
        elif filter_mode == 'median':
            # clip bad pixels
            y_filtered = stats.sigma_clip(y, sigma=3, maxiters=50)
            mask = np.logical_not(np.logical_or(bpm[:,x], y_filtered.mask))
            if np.sum(mask) < 0.5*len(wave_x):
                P[:,x] = np.zeros_like(y)
            else:
                # apply a median filter
                y_smooth = ndimage.median_filter(y[mask], filter_width)
                P[:,x] = interp1d(wave_x[mask], y_smooth, kind='linear', 
                                    bounds_error=False, 
                                    fill_value=np.nanmedian(y_smooth))(wave_x)
                # plt.plot(wave_x[mask], y[mask])
                # plt.plot(wave_x[mask], y_smooth)
                # plt.plot(wave_x, P[:,x])
                # plt.show()

    # ensure the positivity of P, mute the values far away from peak
    psf = np.mean(P, axis=0)
    indice = np.argwhere(psf<=0).T[0]
    if len(indice)>0:
        ind = np.searchsorted(indice, len(psf)//2)
        if ind == 0:
            P[:, indice[0]:] = etol
        elif ind < len(indice) and ind > 0:
            P[:, :indice[ind-1]+1] = etol
            P[:, indice[ind]:] = etol
        elif ind == len(indice):
            P[:, :indice[-1]+1] = etol

    # Normalize the spatial profile per wavelength channel
    for w in wave_x:
        P[w] /= np.sum(P[w])

    # bad pixel map with good pixels = 1 and bad pxiels = 0.
    M_bp = np.ones_like(D, dtype=bool)

    if extr_level is not None:
        psf = np.mean(P, axis=0)
        # determine the extraction aperture by including 95% flux
        cdf = np.cumsum(psf)     
        extr_aper = (cdf > (1.-extr_level)/2.) & (cdf < (1.+extr_level)/2.)
        extr_aper_not = (cdf < (1.-extr_level)/2.) | (cdf > (1.+extr_level)/2.)
        # D[:, extr_aper_not] = 0.
        V_new[:, extr_aper_not] = 1./etol
        P[:, extr_aper_not] = etol
        P[P==0] = etol
        #     plt.plot(cdf)
        #     plt.show()

        # mask bad pixels
        M_bp[:, extr_aper_not] = False
        if np.sum(extr_aper) < 7:
            # when aperture is small, sigma_clip along one axis is not effective
            norm_filtered = stats.sigma_clip((D/P)[:,extr_aper], sigma=4)
        else:
            norm_filtered = stats.sigma_clip((D/P)[:,extr_aper], sigma=4, axis=1)
        M_bp[:, extr_aper] &= np.logical_not(norm_filtered.mask)
        # ax1 = plt.subplot(131)
        # ax2 = plt.subplot(132, sharey=ax1)
        # ax3 = plt.subplot(133, sharey=ax1)
        # ax1.imshow((D)[:,extr_aper], aspect='auto', vmin=0, vmax=2e2)
        # ax2.imshow((D/P)[:,extr_aper], aspect='auto', vmin=0, vmax=50)
        # ax3.imshow(norm_filtered.mask, aspect='auto')
        # plt.show()

    M_bp[:10] = False
    M_bp[-10:] = False

    # if debug:
    #     plt.plot(f_std)

    # iteratively reject bad pixels and update optimal spectrum and variance.
    for ite in range(max_iter):

        f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1) + etol)
        # plt.plot(f_opt)
        V_new = V + np.abs(P*f_opt[:,None]) / gain / NDIT
        # Residual of optimally extracted spectrum and the observation
        Res = M_bp * (D - P*f_opt[:,None])**2/V_new

        # to avoid the Residuals driven by the bad pxiels.
        good_channels = np.all(Res<badpix_clip**2, axis=1)
        # plt.plot(wave_x, good_channels*np.max(f_opt))
        # plt.show()
        f_prox = interp1d(wave_x[good_channels], f_opt[good_channels], 
                    kind='linear', bounds_error=False, fill_value=0.)(wave_x)
        Res = M_bp * (D - P*f_prox[:,None])**2/V_new

        # only reject one bad pixel per wavelength channel at a time
        bad_channels = np.any(Res>badpix_clip**2, axis=1)
        for x in wave_x[bad_channels]:
            M_bp[x, np.argmax(Res[x]-badpix_clip**2)] = False

        # if debug:
        #     # plt.plot(bad_channels.astype(float)*np.median(f_opt))
        #     # plt.show()
        #     fig, axes = plt.subplots(ncols=2, sharey=True)
        #     axes[0].imshow(Res, vmin=0, vmax=40, aspect='auto')
        #     axes[1].imshow(M_bp, aspect='auto')
        #     plt.show()

        if not np.any(bad_channels):
            break

    if debug:
        print(ite)
        plt.plot(f_opt)
        plt.show()

    # Rescale the variance by the reduced chi2
    chi2_r = np.nansum(Res)/(np.sum(M_bp)-len(f_opt))
    
    if chi2_r > 1:
        var = 1. / (np.sum(M_bp*P*P/V_new, axis=1)+etol) * chi2_r
    else:
        var = 1. / (np.sum(M_bp*P*P/V_new, axis=1)+etol)
    
    # Optimally extracted spectrum
    f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1) + etol)

    return f_opt, np.sqrt(var), D.T, V_new.T, P.T



def func_wlen_optimization(poly, *args):
    """
    cost function for optimizing wavelength solutions

    Parameters
    ----------
    poly: array
        polynomial coefficients for correcting the wavelength solution
    args: 
        wave, flux: initial wavelengths and observed flux
        template_interp_func: model spectrum 
    
    Returns
    -------
    correlation: float
        minus correlation between observed and model spectra
    """
    
    wave, flux, template_interp_func = args

    # Apply the polynomial coefficients
    new_wave = Poly.polyval(wave - np.mean(wave), poly) + wave

    # Interpolate the template onto the new wavelength grid
    template = template_interp_func(new_wave)

    # Maximize the cross correlation
    correlation = -template.dot(flux)
    # chi_squared = np.sum((template-flux)**2/err**2)

    return correlation


def wlen_solution(fluxes, errs, w_init, transm_spec, order=2,
                  p_range=[0.5, 0.05, 0.01],
                cont_smooth_len=101,
                debug=False):
    """
    Method for refining wavelength solution using a quadratic 
    polynomial correction Poly(p0, p1, p2). The optimization 
    is achieved by maximizing cross-correlation functions 
    between the spectrum and a telluric transmission model on 
    a order-by-order basis.

    Parameters
    ----------

    fluxes: array
        flux of observed spectrum in each spectral order
    w_init: array
        initial wavelengths of each spectral order
    p0_range: float
        the absolute range of the 0th polynomial coefficient
    p1_range: float
        the absolute range of the 1th polynomial coefficient
    p2_range: float
        the absolute range of the 2th polynomial coefficient
    cont_smooth_len: int
        the window length used in the high-pass filter to remove 
        the continuum of observed spectrum
    debug : bool
        if True, print the best fit polynomial coefficients.

    Returns
    -------
    wlens: array
        the refined wavelength solution
        
    """

    # Prepare a function to interpolate the skycalc transmission
    template_interp_func = interp1d(transm_spec[:,0], transm_spec[:,1], 
                                    kind='linear')
    
    wlens = []
    Ncut = 10
    minimum_strength=0.0005

    for o in range(len(fluxes)):

        f, f_err, wlen_init = fluxes[o], errs[o], w_init[o]

        # ignore the detector-edges 
        f, w, f_err = f[Ncut:-Ncut], wlen_init[Ncut:-Ncut], f_err[Ncut:-Ncut]

        # Remove continuum and nans of spectra.
        # The continuum is estimated by smoothing the
        # spectrum with a 2nd order Savitzky-Golay filter
        nans = np.isnan(f)
        continuum = signal.savgol_filter(
                    f[~nans], window_length=cont_smooth_len,
                    polyorder=2, mode='interp')
        
        f = f[~nans] - continuum
        # outliers = np.abs(f)>(5*np.nanstd(f))
        # f[outliers]=0
        f, w, f_err = f[Ncut:-Ncut], w[~nans][Ncut:-Ncut], f_err[Ncut:-Ncut]

        index_o = (transm_spec[:,0]>np.min(wlen_init)) & \
                  (transm_spec[:,0]<np.max(wlen_init))

        bound = [(-p_range[j], p_range[j]) for j in range(order+1)]
        # Check if there are enough telluric features in this wavelength range
        if np.std(transm_spec[:,1][index_o]) > minimum_strength: 
            
            # Use scipy.optimize to find the best-fitting coefficients
            res = optimize.minimize(
                        func_wlen_optimization, 
                        args=(w, f, template_interp_func), 
                        x0=np.zeros(order+1), method='Nelder-Mead', tol=1e-8, 
                        bounds=bound) 
            poly_opt = res.x

            result = [f'{item:.6f}' for item in poly_opt]

            if debug:
                print(f"Order {o} -> Poly(x^0, x^1, x^2): {result}")

            # if the coefficient hits the prior edge, fitting is unsuccessful
            # fall back to the 0th oder solution.
            if np.isclose(np.abs(poly_opt[-1]), p_range[-1]):
                warnings.warn(f"Fitting of wavelength solution for order {o} is unsuccessful. Only a 0-order offset is applied.")
                res = optimize.minimize(
                        func_wlen_optimization, 
                        args=(w, f, template_interp_func), 
                        x0=[0], method='Nelder-Mead', tol=1e-8, 
                        bounds=[(-p_range[0],+p_range[0]),
                                ])
                poly_opt = res.x
                if debug:
                    print(poly_opt)

            wlen_cal = wlen_init + \
                    Poly.polyval(wlen_init - np.mean(wlen_init), poly_opt)     
        else:
            warnings.warn(f"Not enough telluric features to correct wavelength for order {o}")
            wlen_cal = wlen_init

        wlens.append(wlen_cal)

    return wlens
    # return w_init


def SpecConvolve(in_wlen, in_flux, out_res, in_res=1e6, verbose=False):
    """
    Convolve the input spectrum to a lower resolution with a Gaussian kernel.
    
    Parameters
    ----------

    in_wlen: array 
        input wavelength array 
    in_flux: array
        input flux at high resolution
    out_res: int
        output resolution (low)
    in_res: int 
        input resolution (high) R~w/dw
    verbose: bool
        if True, print out the sigma of Gaussian filter used
    
    Returns
    ----------

    flux_LSF: array
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


def SpecConvolve_GL(in_wlen, in_flux, out_res, gamma, in_res=1e6):
    """
    Convolve the input spectrum to a lower resolution with a Voigt kernel,
    i.e. the convolution of a Gaussian and a Lorentzian kernel.
    
    Parameters
    ----------

    in_wlen: array 
        input wavelength array 
    in_flux: array
        input flux at high resolution
    out_res: int
        output resolution (low)
    gamma: float
        the scale parameter of a Lorentzian profile 
    in_res: int 
        input resolution (high) R~w/dw
    verbose: bool
        if True, print out the sigma of Gaussian filter used
    
    Returns
    ----------

    flux_V: array
        Convolved spectrum
    """

    def G(x, sigma):
        """ Return Gaussian line shape wth sigma """
        return 1./ np.sqrt(2. * np.pi) / sigma\
                                * np.exp(-(x / sigma)**2 / 2.)

    def L(x, gamma):
        """ Return Lorentzian line shape at x with HWHM gamma """
        return gamma / np.pi / (x**2 + gamma**2)

    sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))
    spacing = np.mean(2.*np.diff(in_wlen)/ (in_wlen[1:]+in_wlen[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF/spacing
    # print(sigma_LSF_gauss_filter, out_res, in_res)
    
    xx = np.arange(-int(20*sigma_LSF_gauss_filter), int(20*sigma_LSF_gauss_filter+1), 1)
    win_G = G(xx, sigma_LSF_gauss_filter)
    if np.isclose(gamma, 0):
        win_V = win_G
    else:
        win_L = L(xx, gamma)
        win_V = signal.convolve(win_L, win_G, mode='same') / sum(win_G)
    flux_V = signal.convolve(in_flux, win_V, mode='same') / sum(win_V)

    return flux_V


def PolyfitClip(x, y, order, mask=None, clip=4., max_iter=20):
    """
    Perform weighted least-square polynomial fit,
    iterratively cliping pixels above a certain sigma threshold
    
    Parameters
    ----------

    x, y: array
        the x and y arrays to be fitted 
    order: int
        degree of polynomial 
    clip: int 
        sigma clip threshold
    max_iter: int 
        max number of iteration in sigma clip
    mask: bool
        boolean array with the masked values set to False.

    Returns
    ----------

    y_model: array
        polynomial fitted results
    coeffs: array
        best fit polynomial coefficient
    """

    x_mean = np.array(x) - np.nanmean(x) 
    A_full = np.vander(x_mean, order)

    if mask is None:
        mask = np.ones_like(x_mean, dtype=bool)

    x_use = x_mean[mask]
    y_use = np.array(y)[mask]

    for i in range(max_iter):
        A_matrix = np.vander(x_use, order)
        coeffs = np.linalg.solve(np.dot(A_matrix.T, A_matrix), 
                                 np.dot(A_matrix.T, y_use))
        y_model = np.dot(A_matrix, coeffs)
        res = (y_use - y_model)
        # plt.scatter(x_peaks[mask], res)
        # plt.axhline(-sigma*std)
        if np.any(np.abs(res) > clip*np.std(res)):
            mask = np.ones_like(x_use, dtype=bool)
            mask &= (np.abs(res) < clip*np.std(res))
        # if np.min(np.abs(res)) < clip*np.std(res):
        #     mask[np.argmax(np.abs(res))] = False
        else:
            break
        x_use = x_use[mask]
        y_use = y_use[mask]

    y_model = np.dot(A_full, coeffs)
    # print(coeffs, i)
    final_mask = (np.abs(y - y_model) > clip*np.std(res))

    # ite=0
    # while ite < max_iter:
    #     poly = Poly.polyfit(xx[mask], yy[mask], dg, w=ww[mask])
    #     y_model = Poly.polyval(xx, poly)
    #     res = yy - y_model
    #     threshold = np.std(res[mask])*clip
    #     if plotting:
    #         plt.plot(yy)
    #         plt.plot(y_model)
    #         plt.show()
    #         plt.plot(res)
    #         plt.axhline(threshold)
    #         plt.axhline(-threshold)
    #         plt.ylim((-1.2*threshold,1.2*threshold))
    #         plt.show()
    #     if np.any(np.abs(res[mask]) > threshold):
    #         mask = mask & (np.abs(res) < threshold)
    #     else:
    #         break
    #     ite+=1
    return y_model, coeffs, final_mask



def rot_int_cmj(wave, flux, vsini, epsilon=0.6, nr=10, ntheta=100, dif = 0.0):
	"""

	Adapted from Carvalho & Johns-Krull (2023).
	See https://github.com/Adolfo1519/RotBroadInt

	A routine to quickly rotationally broaden a spectrum in linear time.
	INPUTS:
	s - input spectrum
	w - wavelength scale of the input spectrum
	vsini (km/s) - projected rotational velocity

	OUTPUT:
	ns - a rotationally broadened spectrum on the wavelength scale w
	OPTIONAL INPUTS:
	eps (default = 0.6) - the coefficient of the limb darkening law
	nr (default = 10) - the number of radial bins on the projected disk

	ntheta (default = 100) - the number of azimuthal bins in the largest radial annulus
	note: the number of bins at each r is int(r*ntheta) where r < 1
	
	dif (default = 0) - the differential rotation coefficient, applied according to the law
	Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)). Dif = .675 nicely reproduces the law 
	proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS. Dif = .23 is 
	similar to observed solar differential rotation. Note: the th in the above expression is 
	the stellar co-latitude, not the same as the integration variable used below. This is a 
	disk integration routine.
	"""

	ns = np.copy(flux)*0.0
	tarea = 0.0
	dr = 1./nr
	for j in range(0, nr):
		r = dr/2.0 + j*dr
		area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - epsilon + epsilon*np.cos(np.arcsin(r)))
		for k in range(0,int(ntheta*r)):
			th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
			if dif != 0:
				vl = vsini * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
				ns += area * np.interp(wave + wave*vl/2.99792458e5, wave, flux)
				tarea += area
			else:
				vl = r * vsini * np.sin(th)
				ns += area * np.interp(wave + wave*vl/2.99792458e5, wave, flux)
				tarea += area
		  
	return ns/tarea


def get_PHOENIX_stellar_model(temp, wave_cut, logg=4.0):
    """
    Get the PHOENIX stellar model.

    Parameters
    ----------
    temp: int
        The effective temperature of the standard star.
    wave_cut: list
        The wavelength range to use for the stellar model.
    logg: float
        The surface gravity of the standard star.

    Returns
    -------
    w: array
        The wavelength of the stellar model.
    f: array
        The flux of the stellar model.
    """

    src_path = os.path.dirname(os.path.abspath(__file__))
    wave_file = os.path.join(src_path, '../../data/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    wave = fits.getdata(wave_file) * 1e-1 #nm

    url = 'https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/' 
    filename = f'lte{temp:05d}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    if not os.path.isfile(filename):
        print(f"Downloading PHOENIX stellar model T={temp} K")
        r = requests.get(url+filename)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise RuntimeError(f"Failed to download a Phoenix stellar model for temp={temp}, logg={logg} ({url}): {err}") from err
        with open(filename, "wb") as f:
            f.write(r.content)
        print("[DONE]")
    flux = fits.getdata(filename)  #'erg/s/cm^2/cm'
    indices = (wave > wave_cut[0]) & (wave < wave_cut[1])
    f = flux[indices]/np.median(flux[indices])
    w = wave[indices]

    return w, f


def instrument_response(std, tellu, temp, vsini, vsys=0., 
                        mask_wave=[], debug=False):
    """
    Method for calibrating spectral shape using standard star observations.

    Parameters
    ----------
    std: SPEC
        spectrum of the observed standard star
    tellu: ndarray
        telluric transmission template with wavelength on the first column and transmission on the second  
    temp: int
        effective temperature of the standard star
    vsini: float
        rotational broadenning of the standard star
    vsys: float
        systemic and barycentric  velocity correction for the standard star   
    mask_wave: list of tuple
        list of lower and upper limit for wavelength ranges (in nm) that need to 
        be masked when estimating the isntrument response.
        They are usually Hydrogen lines of the stellar spectrum.
    """

    wave, flux, _ = std.get_spec1d()
    tellu = interp1d(tellu[:,0], tellu[:,1])(wave)
    
    # adjust to be compatible with the Phoenix model
    if temp < 2300 or temp > 12000:
        raise ValueError(f"Standard star temperature {temp} K is out of Phoenix model range (2300-12000 K).")
    modulus = 100 if temp < 7000 else 200
    if temp % modulus != 0:
        warnings.warn(f"Standard star temperature {temp} K is not a multiple of {modulus} K - using the closest Phoenix model ({int(np.round(temp/modulus)*modulus)} K)")
        temp = int(np.round(temp/modulus)*modulus)
        
    # get phoenix stellar model
    wave_model, flux_model = get_PHOENIX_stellar_model(temp, wave_cut=[wave[0]-100, wave[-1]+100])
    # rotationally broaden and rv shift
    flux_model = rot_int_cmj(wave_model, flux_model, vsini)
    w_shift = wave * (1. - vsys/const.c.to("km/s").value)
    flux_model_interp = interp1d(wave_model, flux_model)(w_shift)

    # plt.plot(wave, flux)
    # plt.plot(wave, tellu)
    # plt.plot(wave, flux_model_interp)
    # plt.show()
    
    # instrument response
    transm = flux/flux_model_interp/tellu
    transm = np.reshape(transm, std.wlen.shape)
    flux_tellu = np.reshape(tellu, std.wlen.shape)

    inst_res = []
    for i, x in enumerate(std.wlen):
        y = transm[i]
        mask = (flux_tellu[i] < 0.5) | np.isnan(y)
        y_model, _, _ = PolyfitClip(x, y, order=12, mask=~mask, clip=3, max_iter=50)
        inst_res.append(y_model)
    #     plt.plot(x, y)
    #     plt.plot(x, y_model)
    # plt.show()
    
    # get the detrended telluric spectrum of the standard star
    tellu_clean = flux / np.ravel(inst_res) / flux_model_interp

    if debug:
        plt.plot(wave, tellu_clean)
        plt.plot(wave, tellu)
        plt.show()

    inst_res = flux/tellu_clean/flux_model_interp

    # fit the same polynomial for all three detectors
    std.wlen = std.wlen.reshape((std.wlen.shape[0]//3, -1))
    inst_res = inst_res.reshape(std.wlen.shape)

    resp = []
    for i, x in enumerate(std.wlen):
        y = inst_res[i]
        mask = np.isnan(y)
        for w_range in mask_wave:
            mask = np.logical_or((x > w_range[0]) & (x < w_range[1]), mask)
        y_model, _, _ = PolyfitClip(x, y, order=2, mask=~mask, clip=3)
        resp.append(y_model)
        if debug:
            plt.plot(x, y)
            plt.plot(x, y_model)

    if debug:
        plt.show()

    return np.ravel(resp), tellu_clean


def create_eso_recipe_config(eso_recipe, outpath, verbose):
    """
    https://github.com/tomasstolker/pycrires
    Internal method for creating a configuration file with default
    values for a specified `EsoRex` recipe. 

    Parameters
    ----------
    eso_recipe : str
        Name of the `EsoRex` recipe.
    outpath : str
        Path to save the config files.
    verbose : bool
        Print output produced by ``esorex``.

    Returns
    -------
    NoneType
        None
    """

    config_file = os.path.join(outpath, f"{eso_recipe}.rc") 

    if shutil.which("esorex") is None:
        raise RuntimeError(
            "Esorex is not accessible from the command line. "
            "Please make sure that the ESO pipeline is correctly "
            "installed and included in the PATH variable."
        )

    if not os.path.exists(config_file):
    # if True:
        print()

        esorex = ["esorex", f"--create-config={config_file}", eso_recipe]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=outpath, stdout=stdout, check=True)

        # Open config file and adjust some parameters
        with open(config_file, "r", encoding="utf-8") as open_config:
            config_text = open_config.read()

        if eso_recipe == "molecfit_model":
            config_text = config_text.replace(
                "USE_INPUT_KERNEL=TRUE",
                "USE_INPUT_KERNEL=FALSE",
            )

            config_text = config_text.replace(
                "LIST_MOLEC=NULL",
                "LIST_MOLEC=H2O,CO2,CO,CH4,O2,N2O",
            )

            config_text = config_text.replace(
                "FIT_MOLEC=NULL",
                "FIT_MOLEC=1,1,0,1,0,1",
            )

            config_text = config_text.replace(
                "REL_COL=NULL",
                "REL_COL=1.0,1.0,1.0,1.0,1.0,1.0",
            )

            config_text = config_text.replace(
                "MAP_REGIONS_TO_CHIP=1",
                "MAP_REGIONS_TO_CHIP=NULL",
            )

            config_text = config_text.replace(
                "COLUMN_LAMBDA=lambda",
                "COLUMN_LAMBDA=WAVE",
            )

            config_text = config_text.replace(
                "COLUMN_FLUX=flux",
                "COLUMN_FLUX=FLUX",
            )

            config_text = config_text.replace(
                "COLUMN_DFLUX=NULL",
                "COLUMN_DFLUX=FLUX_ERR",
            )

            config_text = config_text.replace(
                "PIX_SCALE_VALUE=0.086",
                "PIX_SCALE_VALUE=0.056",
            )

            config_text = config_text.replace(
                "FTOL=1e-10",
                "FTOL=1e-8",
            )

            config_text = config_text.replace(
                "XTOL=1e-10",
                "XTOL=1e-8",
            )

            config_text = config_text.replace(
                "CHIP_EXTENSIONS=FALSE",
                "CHIP_EXTENSIONS=TRUE",
            )

            config_text = config_text.replace(
                "FIT_WLC=0",
                "FIT_WLC=NULL",
            )

            config_text = config_text.replace(
                "WLC_N=1",
                "WLC_N=1",
            )

            config_text = config_text.replace(
                "WLC_CONST=-0.05",
                "WLC_CONST=0.0",
            )

            config_text = config_text.replace(
                "FIT_CONTINUUM=1",
                "FIT_CONTINUUM=NULL",
            )

            config_text = config_text.replace(
                "CONTINUUM_N=0",
                "CONTINUUM_N=NULL",
            )

            config_text = config_text.replace(
                "FIT_RES_BOX=TRUE",
                "FIT_RES_BOX=FALSE",
            )

            config_text = config_text.replace(
                "RES_BOX=1.0",
                "RES_BOX=0",
            )

            config_text = config_text.replace(
                "RES_GAUSS=1.0",
                "RES_GAUSS=3.0",
            )

            config_text = config_text.replace(
                "RES_LORENTZ=1.0",
                "RES_LORENTZ=0.5",
            )

            config_text = config_text.replace(
                "FIT_RES_LORENTZ=TRUE",
                "FIT_RES_LORENTZ=FALSE",
            )

            config_text = config_text.replace(
                "KERNMODE=FALSE",
                "KERNMODE=TRUE",
            )

            config_text = config_text.replace(
                "VARKERN=FALSE",
                "VARKERN=TRUE",
            )


        elif eso_recipe == "molecfit_calctrans":
            config_text = config_text.replace(
                "USE_INPUT_KERNEL=TRUE",
                "USE_INPUT_KERNEL=FALSE",
            )

            config_text = config_text.replace(
                "CHIP_EXTENSIONS=FALSE",
                "CHIP_EXTENSIONS=TRUE",
            )

        elif eso_recipe == "molecfit_correct":
            config_text = config_text.replace(
                "CHIP_EXTENSIONS=FALSE",
                "CHIP_EXTENSIONS=TRUE",
            )

        with open(config_file, "w", encoding="utf-8") as open_config:
            open_config.write(config_text)

        if not verbose:
            print(" [DONE]")


def molecfit(input_path, spec, wave_range=None, savename=None, verbose=False):
    """
    A wrapper of molecfit for telluric correction

    Parameters
    ----------
    input_path : str
        the working directory for molecfit.
    spec: SPEC
        input spectra for molecfit
    wave_range: list of tuple
        list of wavelength regions to be indcluded for the telluric fitting
    savename : str
        the filename for the input files and fitting results.
    verbose : bool
        Print output produced by ``esorex``.

    Returns
    -------
    NoneType
        None
    """
    
    if savename is None:
        savename = "SCIENCE.fits"

    primary_hdu = fits.PrimaryHDU(header=spec.header)

    hdul_out = fits.HDUList([primary_hdu])

    w0 = [spec.wlen[i][~np.isnan(spec.flux[i])][0] for i in range(len(spec.wlen))]
    w1 = [spec.wlen[i][~np.isnan(spec.flux[i])][-1] for i in range(len(spec.wlen))]
    if wave_range is None:
        wmin, wmax = w0, w1
        map_chip = range(spec.wlen.shape[0])
    else:
        mask = np.zeros_like(spec.wlen)
        for w_range in wave_range:
            mask = np.logical_or((spec.wlen > w_range[0]) & 
                                 (spec.wlen < w_range[1]), mask)
        mask &= (~np.isnan(spec.flux))
        # find min and max wavelength of each region
        w_masked = spec.wlen[mask]
        indice_split = np.diff(w_masked) > 10*np.max(np.diff(spec.wlen))
        wmin = w_masked[np.append(True, indice_split)]
        wmax = w_masked[np.append(indice_split, True)]

        # map wavelength ranges to chips
        map_chip = [np.searchsorted(w1, a) for a in wmin]
    
    # define flags for wavelength solution and continuum fitting
    map_ext = range(0, spec.wlen.shape[0]+1)
    wlc_fit = np.ones_like(map_chip, dtype=int) - 1
    cont_fit = np.ones_like(map_chip, dtype=int)
    cont_fit_poly = np.ones_like(map_chip, dtype=int)

    # Loop over chips
    for i_det in range(spec.wlen.shape[0]):
        wave = spec.wlen[i_det]
        flux = spec.flux[i_det]#/np.nanmedian(spec.flux[i_det])
        flux_err = spec.err[i_det]#/np.nanmedian(spec.flux[i_det])
        if verbose:   
            plt.plot(wave, flux, color='k', alpha=0.7)
        mask = np.isnan(flux) | np.isnan(flux_err)
        col1 = fits.Column(name='WAVE', format='D', array=wave)
        col2 = fits.Column(name='FLUX', format='D', array=flux)
        col3 = fits.Column(name='FLUX_ERR', format='D', array=flux_err)
        table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
        hdul_out.append(table_hdu)
    
    if verbose:        
        for a,b in zip(wmin, wmax): 
            plt.axvspan(a, b, alpha=0.3, color='r')
        plt.show()

    # Create FITS file with SCIENCE
    file_science = os.path.join(input_path, savename)
    hdul_out.writeto(file_science, overwrite=True)

    # Create FITS file with WAVE_INCLUDE
    col_wmin = fits.Column(name="LOWER_LIMIT", format="D", array=wmin)
    col_wmax = fits.Column(name="UPPER_LIMIT", format="D", array=wmax)
    col_map = fits.Column(name="MAPPED_TO_CHIP", format="I", array=map_chip)
    col_cont = fits.Column(name="CONT_FIT_FLAG", format="I", array=cont_fit)
    col_cont_poly = fits.Column(name="CONT_POLY_ORDER", format="I", array=cont_fit_poly)
    col_wlc = fits.Column(name="WLC_FIT_FLAG", format="I", array=wlc_fit)
    columns = [col_wmin, col_wmax, col_map, col_cont, col_cont_poly, col_wlc]
    table_hdu = fits.BinTableHDU.from_columns(columns)
    file_wave_inc = os.path.join(input_path, "WAVE_INCLUDE.fits")
    table_hdu.writeto(file_wave_inc, overwrite=True)

    # Create FITS file with MAPPING_ATMOSPHERIC
    name = "ATM_PARAMETERS_EXT"
    col_atm = fits.Column(name=name, format="K", array=map_ext)
    table_hdu = fits.BinTableHDU.from_columns([col_atm])
    file_map_atm = os.path.join(input_path, "MAPPING_ATMOSPHERIC.fits")
    table_hdu.writeto(file_map_atm, overwrite=True)

    # Create FITS file with MAPPING_CONVOLVE
    name = "LBLRTM_RESULTS_EXT"
    col_conv = fits.Column(name=name, format="K", array=map_ext)
    table_hdu = fits.BinTableHDU.from_columns([col_conv])
    file_map_conv = os.path.join(input_path, "MAPPING_CONVOLVE.fits")
    table_hdu.writeto(file_map_conv, overwrite=True)

    # Create FITS file with MAPPING_CORRECT
    name = "TELLURIC_CORR_EXT"
    col_corr = fits.Column(name=name, format="K", array=map_ext)
    table_hdu = fits.BinTableHDU.from_columns([col_corr])
    file_map_corr = os.path.join(input_path, "MAPPING_CORRECT.fits")
    table_hdu.writeto(file_map_corr, overwrite=True)

    # prepare sof and run molecfit_model with esorex
    sof_file = os.path.join(input_path, "model.sof")
    with open(sof_file, "w", encoding="utf-8") as sof_open:
        sof_open.write(f'{file_science} SCIENCE\n')
        sof_open.write(f'{file_wave_inc} WAVE_INCLUDE\n')

    create_eso_recipe_config("molecfit_model", input_path, verbose=verbose)
    config_file = os.path.join(input_path, "molecfit_model.rc")

    esorex = [
        "esorex",
        f"--recipe-config={config_file}",
        f"--output-dir={input_path}",
        "molecfit_model",
        sof_file,
    ]

    if verbose:
        stdout = None
    else:
        stdout = subprocess.DEVNULL
        print("Running EsoRex...", end="", flush=True)

    subprocess.run(esorex, cwd=input_path, stdout=stdout, check=True)

    if not verbose:
        print(" [DONE]")

    file_atm_par = os.path.join(input_path, "ATM_PARAMETERS.fits")
    file_mol = os.path.join(input_path, "MODEL_MOLECULES.fits")
    file_best_par = os.path.join(input_path, "BEST_FIT_PARAMETERS.fits")
    # prepare sof and run molecfit_calctrans with esorex
    sof_file = os.path.join(input_path, "calctrans.sof")
    with open(sof_file, "w", encoding="utf-8") as sof_open:
        sof_open.write(f'{file_science} SCIENCE\n')
        sof_open.write(f'{file_map_atm} MAPPING_ATMOSPHERIC\n')
        sof_open.write(f'{file_map_conv} MAPPING_CONVOLVE\n')
        sof_open.write(f'{file_atm_par} ATM_PARAMETERS\n')
        sof_open.write(f'{file_mol} MODEL_MOLECULES\n')
        sof_open.write(f'{file_best_par} BEST_FIT_PARAMETERS\n')

    create_eso_recipe_config("molecfit_calctrans", input_path, verbose=verbose)
    config_file = os.path.join(input_path, "molecfit_calctrans.rc")

    esorex = [
        "esorex",
        f"--recipe-config={config_file}",
        f"--output-dir={input_path}",
        "molecfit_calctrans",
        sof_file,
    ]

    if verbose:
        stdout = None
    else:
        stdout = subprocess.DEVNULL
        print("Running EsoRex...", end="", flush=True)

    subprocess.run(esorex, cwd=input_path, stdout=stdout, check=True)

    if not verbose:
        print(" [DONE]")

    file_tellu_corr = os.path.join(input_path, "TELLURIC_CORR.fits")
    # prepare sof and run molecfit_model with esorex
    sof_file = os.path.join(input_path, "correct.sof")
    with open(sof_file, "w", encoding="utf-8") as sof_open:
        sof_open.write(f'{file_science} SCIENCE\n')
        sof_open.write(f"{file_map_corr} MAPPING_CORRECT\n")
        sof_open.write(f"{file_tellu_corr} TELLURIC_CORR\n")

    create_eso_recipe_config("molecfit_correct", input_path, verbose=verbose)
    config_file = os.path.join(input_path, "molecfit_correct.rc")

    esorex = [
        "esorex",
        f"--recipe-config={config_file}",
        f"--output-dir={input_path}",
        "molecfit_correct",
        sof_file,
    ]

    if verbose:
        stdout = None
    else:
        stdout = subprocess.DEVNULL
        print("Running EsoRex...", end="", flush=True)

    subprocess.run(esorex, cwd=input_path, stdout=stdout, check=True)

    if not verbose:
        print(" [DONE]")


# Sage Santomenna
def get_star_properties(obj_name:str):
    """Look up the teff, vsini, and rv of a star by name
    
    Parameters
    ----------
    obj_name: str
        the name of the standard star, e.g. '10 Lep'

    Returns
    ----------
    teff: float
        the effective temperature of the star in Kelvin
    vsini: float
        the projected rotational velocity of the star in km/s
    rv: float
        systemic and barycentric velocity correction for the standard star in km/s
    """
    from astroquery.simbad import Simbad
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields("rvz_radvel")
    basic_q = Simbad.query_object(obj_name)
    if not len(basic_q):
        raise ValueError(f"No results found for the object {obj_name} in Simbad.")
    basic_q = basic_q[~basic_q["rvz_radvel"].mask]
    if not len(basic_q):
        raise ValueError(f"No radial velocity found for the object {obj_name} in Simbad.")
    rv = basic_q["rvz_radvel"][0]
    
    Simbad.add_votable_fields("mesFe_h")
    temp_q = Simbad.query_object(obj_name)
    if not len(temp_q):
        raise ValueError(f"No temperature found for the object {obj_name} in Simbad.")
    temp_q = temp_q[~temp_q["mesfe_h.teff"].mask]
    if not len(temp_q):
        raise ValueError(f"No temperature found for the object {obj_name} in Simbad.")
    temp_q.sort("mesfe_h.mespos") # get best
    teff = temp_q["mesfe_h.teff"][0]
    
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields("mesRot")
    vsini_q = Simbad.query_object(obj_name)
    if not len(vsini_q):
        raise ValueError(f"No vsini found for the object {obj_name} in Simbad.")
    vsini_q = vsini_q[~vsini_q["mesrot.vsini"].mask]
    if not len(vsini_q):
        raise ValueError(f"No vsini found for the object {obj_name} in Simbad.")
    vsini_q.sort("mesrot.mespos")   # get best  
    vsini = vsini_q["mesrot.vsini"][0]
    
    Simbad.reset_votable_fields()
    return teff, vsini, rv
    
    

