# File: src/excalibuhr/utils.py
__all__ = [
    "combine_frames",
    "detector_shotnoise", 
    "peak_slit_fraction", "align_jitter", "util_unravel_spec",
    "order_trace", "slit_curve", "spectral_rectify_interp",
    "trace_rectify_interp", "mean_collapse", "blaze_norm",
    "readout_artifact", "extract_spec", "remove_starlight",
    "optimal_extraction", "wlen_solution", "xcor_wlen_solution",
    ]

import numpy as np
from astropy.io import fits
from astropy import stats
# from astropy.modeling import models, fitting
from numpy.polynomial import polynomial as Poly
from scipy import ndimage, signal, optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
import warnings


def wfits(fname, im, hdr=None, ext_list= None):
    """
    write data to FITS primary and extensions, overwriting any old file
    
    Parameters
    ----------
    fname: str
        path and filename to which the data is saved 
    im: array
        data to be saved to a FITS file
    hdr: FITS `header`
        header information to be saved
    ext_list: tuple, array, or list
        data to be saved to the extension of the FITS file. 
        It can also be a tuple of data. 

    Returns
    -------
    NoneType
        None
    """

    primary_hdu = fits.PrimaryHDU(im, header=hdr)
    new_hdul = fits.HDUList([primary_hdu])
    if not ext_list is None:
        if isinstance(ext_list, tuple):
            for ext in ext_list:
                new_hdul.append(fits.PrimaryHDU(ext))
        else:
            new_hdul.append(fits.PrimaryHDU(ext_list))
    new_hdul.writeto(fname, overwrite=True, output_verify='ignore') 


def util_master_dark(dt, collapse='median', badpix_clip=5):
    """
    combine dark frames; generate bad pixel map and readout noise frame
    
    Parameters
    ----------
    dt: list
        a list of dark frames to be combined 
    collapse: str
        the way of combining dark frames: `mean` or `median`
    badpix_clip : int
        sigma of bad pixel clipping

    Returns
    -------
    master, rons, badpix: array
        combined dark frame, readout noise frame, and bad pixel map
    """

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
        filtered_data = stats.sigma_clip(det, sigma=badpix_clip)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
    
    return master, rons, badpix


def util_master_flat(dt, dark, collapse='median', badpix_clip=5):
    """
    combine flat frames; generate bad pixel map
    
    Parameters
    ----------
    dt: list
        a list of flat frames to be combined 
    dark: array
        dark frame to be subtracted from the flat 
    collapse: str
        the way of combining flat frames: `mean` or `median`
    badpix_clip : int
        sigma of bad pixel clipping

    Returns
    -------
    master, badpix: array
        combined flat frame and bad pixel map
    """

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
            filtered_data = stats.sigma_clip(det, sigma=badpix_clip)
        badpix[i] = filtered_data.mask
        master[i][badpix[i]] = np.nan
        # plt.imshow(badpix[i])
        # plt.show()

    return master, badpix

def combine_frames(dt, err, collapse='mean'):
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
    im_corr[~badpix] /= flat[~badpix]
    err_corr[~badpix] /= flat[~badpix]

    return im_corr, err_corr


def extract_spec(det, det_err, badpix, trace, slit_meta, blaze, spec_star, 
                    gain, NDIT=1, cen0=90, companion_sep=None, aper_half=20, 
                    bkg_subtract=False, debug=False):
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
    tw: array
        polynomials that delineate the edge of the specific order 
    slit: array
        polynomials that describing the slit curvature 
        as a function of the dispersion axis
    blazes: array
        1D blaze function of each order
    gain: float
        detector gain
    NDIT: int
        number of DIT exposure coadded
    f0: float
        location of the star on slit in pixel 
    f1: float
        location of the companion on slit in pixel
    aper_half: int
        half of extraction aperture in pixel
    bkg_subtract: bool
        in case of sub-stellar companion extraction, whether remove the 
        starlight contamination.
    spec_star: array
        in case of sub-stellar companion extraction and `bkg_subtract=True`,
        provide the extracted stellar spectra. This will be scaled according  
        to the PSF and then removed from the science image before extracting
        the companion spectra.

    Returns
    -------
    f_opt, f_err: array
        the extracted fluxes and their uncertainties
    D, chi2_r: array
        modeled slit function and reduced chi2 of the model (for plotting)
    """

    # Correct for the slit curvature and trace curvature
    dt_rect = spectral_rectify_interp([det, det_err],  
                            badpix, trace, slit_meta, debug=False)
    dt_rect = trace_rectify_interp(dt_rect, trace, debug=False) 

    im, im_err = dt_rect

    xx_grid = np.arange(im.shape[1])
    yy_trace = trace_polyval(xx_grid, trace)
    trace_lower, trace_upper = yy_trace

    flux, err, D, chi2 = [],[],[],[]
    for o, (yy_upper, yy_lower) in enumerate(zip(trace_upper, trace_lower)):
        # Crop out the order from the frame
        # slit_len = np.mean(yy_upper-yy_lower)
        yy_upper = yy_upper[len(xx_grid)//2]
        yy_lower = yy_lower[len(xx_grid)//2]
        im_sub = im[int(yy_lower):int(yy_upper+1)]
        im_err_sub = im_err[int(yy_lower):int(yy_upper+1)]
        bpm_sub = badpix[int(yy_lower):int(yy_upper+1)]

        if cen0 is None:
            # find out the location of the peak signal
            profile = np.nanmedian(im_sub, axis=1)
            cen0 = np.argmax(profile)

        # remove starlight contamination
        if bkg_subtract and companion_sep is not None:
            im_sub, im_err_sub = remove_starlight(im_sub, im_err_sub**2, 
                            spec_star[o]/blaze[o], cen0, cen0-companion_sep, 
                            gain=gain, debug=debug)

        # Pixel-location of target
        obj_cen = cen0
        if companion_sep is not None:
            obj_cen = cen0-companion_sep
        
        # Extract a 1D spectrum using the optimal extraction algorithm
        f_opt, f_err, D_sub, chi2_r = optimal_extraction(
                                im_sub.T, im_err_sub.T**2, bpm_sub.T, 
                                obj_cen=int(np.round(obj_cen)), 
                                aper_half=aper_half, 
                                gain=gain, NDIT=NDIT, debug=debug) 
        flux.append(f_opt/blaze[o])
        err.append(f_err/blaze[o])
        D.append(D_sub)
        chi2.append(chi2_r)

    return flux, err, D, chi2 


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
        (`frac`=0 at the bottom, and 1 at the top) 
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


def order_trace(det, badpix, slitlen, sub_factor=128):
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
    
    Returns
    -------
    [poly_upper, poly_lower]
        The polynomial coeffiences of upper and lower trace of each order
    """
    
    im = np.ma.masked_array(det, mask=badpix)

    # Sub-sample the image along the dispersion axis
    xx = np.arange(im.shape[1])
    xx_bin = xx[::sub_factor] + (sub_factor-1)/2.
    im_bin = np.ma.median(im.reshape(im.shape[0], 
                                     im.shape[1]//sub_factor, 
                                     sub_factor), 
                          axis=2)

    # Subtract a shifted image from its un-shifted self
    # The result is approximately the trace edge
    im_grad = np.abs(im_bin[1:,:]-im_bin[:-1,:])

    # Set insignificant signal (<3sigma) to 0, only peaks are left 
    cont_std = np.nanstd(im_grad, axis=0)
    im_grad[(im_grad < cont_std*3)] = 0
    im_grad = np.nan_to_num(im_grad.data)

    width = 2
    xx_loc, upper, lower   = [], [], []
    # Loop over each column in the sub-sampled image
    for i in range(im_grad.shape[1]):
        yy = im_grad[:,int(i)]

        # Find the pixels where the signal is significant
        indices = signal.argrelmax(yy)[0]

        if len(indices)%2!=0:
            raise RuntimeError("Data are too noisy to clearly identify edges. " 
                          "Please increase `sub_factor`.")

        # Find the y-coordinates of the edges, weighted by the 
        # significance of the signal (i.e. center-of-mass)
        cens = np.array([np.sum(xx[int(p-width):int(p+width+1)]* \
                                yy[int(p-width):int(p+width+1)]) / \
                         np.sum(yy[int(p-width):int(p+width+1)]) \
                         for p in indices])

        # Order of y-coordinates is lower, upper, lower, ...
        lower.append(cens[::2])
        upper.append(cens[1::2])

        # Real x-coordinate of this column
        xx_loc.append(xx_bin[i])

    upper = np.array(upper).T
    lower = np.array(lower).T
    poly_upper, poly_lower = [], []
    poly_order = 2
    order_length_min = 125
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

        if slit_len.min() < order_length_min:
            # skip the order that is incomplete
            continue
        elif slit_len.max()-slit_len.min() > 2.:
            # the upper or lower trace hits the edge.
            if np.mean(yy_up) > im.shape[0]*0.5:
                # refine the upper trace solution with fixed slit length
                yy_up = yy_low + slitlen
                poly_up = Poly.polyfit(xx, yy_up, poly_order)
            else:
                # refine the lower trace solution
                yy_low = yy_up - slit_len
                poly_low = Poly.polyfit(xx, yy_low, poly_order)
            poly_upper.append(poly_up)
            poly_lower.append(poly_low)
        else:
            # refine the trace solution by fixing the slit length
            yy_up = yy_low + slitlen
            yy_mid_new = (yy_up + yy_low) / 2.
            yy_up -= yy_mid_new[len(yy_mid)//2]-yy_mid[len(yy_mid)//2]
            yy_low -= yy_mid_new[len(yy_mid)//2]-yy_mid[len(yy_mid)//2]
            poly_up = Poly.polyfit(xx, yy_up, poly_order)
            poly_low = Poly.polyfit(xx, yy_low, poly_order)
            poly_upper.append(poly_up)
            poly_lower.append(poly_low)

    print(f"-> {len(poly_upper)} orders identified")

    return [poly_lower, poly_upper]


def slit_curve(det, badpix, trace, wlen_min, wlen_max, sub_factor=4, debug=False):
    """
    Trace the spectral orders

    Parameters
    ----------
    det: array
        input fpet image
    badpix: array
        bad pixel map corresponding to the `det` image
    wlen_min, wlen_max: array
        the min and max wavelngth of each order 
        for an initial wavelength solution 
    sub_factor: int
        binning factor along the cross-dispersion axis
    
    Returns
    -------
    [meta0, meta1, meta2]
        The quadratic polynomial coeffiences of the slit tilt 
        as a function of the dispersion axis for each order
    wlen
        initial wavelength solution of each order
    """

    poly_order = 1
    spacing = 40 # minimum spacing (in pixel) of fpet lines
    
    badpix = badpix | np.isnan(det)
    im = np.ma.masked_array(det, mask=badpix)

    xx = np.arange(im.shape[1])
    polys_middle = np.sum(trace, axis=0)/2.
    # Evaluate the upper and lower edges of all orders
    yy_trace = trace_polyval(xx, trace)
    trace_lower, trace_upper = yy_trace

    meta, wlen, x_fpet = [], [], []
    # Loop over each order
    for (yy_upper, yy_lower, middle, w_min, w_max) in \
        zip(trace_upper, trace_lower, polys_middle, wlen_min, wlen_max):

        # Loop over each pixel-row in a sub-sampled image
        slit_image, x_slit, poly_slit = [], [], []
        N_lines = 0
        for row in range(int(yy_lower.min()+1), 
                         int(yy_upper.max()), 
                         sub_factor):

            # combine a few rows (N=sub_factor) to increase 
            # S/N and reject outliers
            im_masked = stats.sigma_clip(im[row:row+sub_factor], 
                                sigma=2, axis=0, masked=True)
            spec = np.ma.mean(im_masked, axis=0)
            mask = (np.sum(im_masked.mask, axis=0) > 0.7*sub_factor)
            badchannel = np.argwhere(mask)[:,0]
            # mask neighboring pixels as well
            badchannel = np.concatenate((badchannel, 
                            badchannel+1, badchannel-1), axis=None)

            # measure the baseline of the spec to be 10% lowest values
            height = np.median(np.sort(spec.data[~mask])[:len(spec)//10])

            # Find the peak pixels (along horizontal axis) 
            peaks, properties = signal.find_peaks(spec, distance=spacing, 
                                        width=10, height=2*height) 
            # mask the peaks identified due to bad channels
            peaks = np.array([item for item in peaks if not item in badchannel])

            # leave out lines around detector edge
            width = np.median(properties['widths'])
            peaks = peaks[(peaks<(im.shape[1]-width)) & (peaks>(width))]

            # Calculate center-of-mass of the peaks
            cens = [np.sum(xx[int(p-width):int(p+width)]* \
                           spec[int(p-width):int(p+width)]) / \
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
            ys = slit_image[:,1][indices == i] # y-coordinate 
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

        # Fit a polynomial to the polynomial coefficients
        # using the x-coordinates of the fpet signal
        poly_slit = np.array(poly_slit)
        meta.append([Poly.polyfit(x_slit, poly_slit[:,i], 2) 
                        for i in range(poly_order+1)])
        x_fpet.append(x_slit)
        
        # Determine the wavelength solution along the detectors/orders
        # Mapping from measured fpet positions to a linear-spaced grid, 
        # fitting with 2nd-order poly
        ii = np.arange(len(x_slit))
        _, poly = PolyfitClip(x_slit, ii, 2)
        
        # Fix the end of the wavelength grid to be values from the header
        grid_poly = Poly.polyfit([Poly.polyval(xx, poly)[0], 
                                  Poly.polyval(xx, poly)[-1]], 
                                 [w_min, w_max], 1)
        ww_cal = Poly.polyval(Poly.polyval(xx, poly), grid_poly)
        wlen.append(ww_cal)

        # w_slit = Poly.polyval(Poly.polyval(x_slit, poly), grid_poly)
        # plt.plot(np.diff(w_slit))
        # plt.show()

    # check rectified image
    if debug:
        spectral_rectify_interp(im, badpix, trace, meta, debug=debug)

    return meta, x_fpet, wlen


def trace_polyval(xx, tw, ):
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
        input image or a list of images to be corrected
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

    bpm = (badpix.astype(bool) | np.isnan(im_rect_spec[0]))
    # bpm_interp = np.zeros_like(bpm).astype(float)

    # Evaluate order traces and slit curvature
    xx_grid = np.arange(0, im_rect_spec.shape[-1])
    yy_trace = trace_polyval(xx_grid, trace)
    trace_lower, trace_upper = yy_trace
    slit_poly = slit_polyval(xx_grid, slit_meta)

    # Loop over each order
    for (yy_upper, yy_lower, poly_full) in zip(trace_upper, trace_lower, slit_poly):
        yy_grid = np.arange(int(yy_lower.min()), int(yy_upper.max()+1))

        isowlen_grid = np.zeros((len(yy_grid), len(xx_grid)))
        # Loop over the horizontal axis
        for x in range(len(xx_grid)):
            isowlen_grid[:, x] = Poly.polyval(yy_grid, poly_full[x])

        # Loop over each row in the order
        for i, (x_isowlen, mask) in enumerate(zip(isowlen_grid, 
                                                bpm[yy_grid])):
            if np.sum(mask)>0.5*len(mask):
                continue
            data_row = im_rect_spec[:, int(yy_grid[0]+i), :]
            
            # Correct for the slit-curvature by interpolating onto the grid
            for r, dt in enumerate(data_row):
                if reverse:
                    im_rect_spec[r, int(yy_grid[0]+i)] = interp1d(x_isowlen[~mask], 
                                                        dt[~mask], 
                                                        kind='cubic', 
                                                        bounds_error=False, 
                                                        fill_value=np.nan
                                                        )(xx_grid)
                else:
                    im_rect_spec[r, int(yy_grid[0]+i)] = interp1d(xx_grid[~mask], 
                                                        dt[~mask], 
                                                        kind='cubic', 
                                                        bounds_error=False, 
                                                        fill_value=np.nan
                                                        )(x_isowlen)
                                                       
    if debug:
        plt.imshow(im_rect_spec[0])
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
            mask = np.isnan(data_col[0]).astype(bool)
            if np.sum(mask)>0.5*len(mask):
                continue 
            for r, dt in enumerate(data_col):
                im_rect[r,yy_grid,x] = interp1d(yy_grid[~mask], dt[~mask], 
                                                kind='cubic', 
                                                bounds_error=False, 
                                                fill_value=np.nan
                                                )(yy_grid+shifts[x])
    if debug:
        plt.imshow(np.log(im_rect[0]))
        plt.show()
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

    # Correct for the slit curvature
    det_rect = spectral_rectify_interp(det, badpix, trace, slit_meta)
    filtered_data = stats.sigma_clip(det_rect, sigma=5)
    badpix_rect = filtered_data.mask | badpix
    # measure the trace again
    trace_update = order_trace(det_rect, badpix_rect, slitlen=slitlen)

    # Retrieve the blaze function by mean-collapsing 
    # the master flat along the slit
    blaze, blaze_image = extract_blaze(det_rect, badpix_rect, trace_update)

    blaze_image = spectral_rectify_interp(blaze_image, np.zeros_like(badpix), 
                        trace_update, slit_meta, reverse=True)
    
    # Set low signal to NaN
    flat_norm = det / blaze_image
    flat_norm[flat_norm<0.5] = np.nan
    flat_norm[flat_norm>1.2] = np.nan

    if debug:
        plt.imshow(flat_norm, vmin=0.8, vmax=1.2)
        plt.show()

    return flat_norm, blaze, trace_update


def extract_blaze(im, badpix, trace, f0=0.5, fw=0.5, sigma=5):
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
        for i in range(len(indices)):
            mask = badpix[indices[i],i] | np.isnan(im[indices[i],i])
            if np.sum(mask)>0.9*len(mask):
                blaze[i] = np.nan
            else:
                blaze[i], _, _ = stats.sigma_clipped_stats(
                    im[indices[i],i][~mask], sigma=sigma)
            blaze_image[indices[i],i] = blaze[i]
        blaze_orders.append(blaze)

    return blaze_orders, blaze_image
    


def readout_artifact(det, det_err, badpix, trace, Nborder=10, sigma=3, debug=False):
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


def remove_starlight(D_full, V_full, spec_star, cen0_p, cen1_p, 
                    aper0=50, aper1=10, aper2=20, sub_factor=32, 
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
        in case of sub-stellar companion extraction and `bkg_subtract=True`,
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

    polys = []
    x_sub = np.arange(0, D_full.shape[-1], sub_factor) + sub_factor/2.
    D = np.reshape(D_full, (D_full.shape[0], 
                            D_full.shape[-1]//sub_factor, 
                            sub_factor))
    D_sub = np.nanmedian(D, axis=2)

    # estimate center of the PSF peak
    # aper2: the window (half width) to determine the line center;
    #        better to be large but, not too large to avoid the companion
    profile = np.nanmedian(-D_sub, axis=1)
    p = np.argmax(profile)
    cen0_n = np.sum(spatial_x[int(p-aper2):int(p+aper2)] * \
                    profile[int(p-aper2):int(p+aper2)]) / \
                    np.sum(profile[int(p-aper2):int(p+aper2)]) 
    cen1_n = cen0_n - cen0_p + cen1_p
    profile = np.nanmedian(D_sub, axis=1)
    p = np.argmax(profile)
    cen0_p = np.sum(spatial_x[int(p-aper2):int(p+aper2)] * \
                    profile[int(p-aper2):int(p+aper2)]) / \
                    np.sum(profile[int(p-aper2):int(p+aper2)]) 

    # aper0: mask the negative primary trace
    # aper1: mask the negative secondary trace and the primary line core
    mask =  ((spatial_x>cen1_n-aper1) & (spatial_x<cen1_n+aper1)) | \
            ((spatial_x>cen0_n-aper0) & (spatial_x<cen0_n+aper0))  
    
    # check at which side the secondary is located
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
        plt.imshow(D_full-bkg, vmin=-10, vmax=20, aspect='auto')
        plt.show()

    return D_full-bkg, Err_full



def optimal_extraction(D_full, V_full, bpm_full, obj_cen, aper_half, \
                       badpix_clip=3, max_iter=10, \
                       gain=2., NDIT=1., etol=1e-6, debug=False):
    """
    Optimal extraction based on Horne(1986)

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

    # Sigma-clip the observation and add bad-pixels to map
    filtered_D = stats.sigma_clip(D, sigma=badpix_clip, axis=1)
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

    # Normalize the polynomial model (profile) per pixel-column
    for w in wave_x:
        P[w] /= np.sum(P[w])

    ite = 0
    M_bp = np.copy(M_bp_init)
    V_new = V + np.abs(D) / gain / NDIT
    while ite < max_iter:
        # Optimally extracted spectrum, obtained by accounting
        # for the profile and variance
        f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1) + etol)
        
        # M_bp[f_opt<0] = False

        # Residual of expanded optimally extracted spectrum and the observation
        Res = M_bp * (D - P*np.tile(f_opt, (P.shape[1],1)).T)**2/V_new

        # Calculate a new variance with the optimally extracted spectrum
        V_new = V + np.abs(P*np.tile(np.abs(f_opt), (P.shape[1],1)).T) / gain / NDIT

        # Halt iterations if residuals are small enough
        if np.all(Res < badpix_clip**2):
            break
        
        for x in wave_x:
            # reject only one outlier at a time.
            if np.any(Res[x]>badpix_clip**2):
                M_bp[x, np.argmax(Res[x]-badpix_clip**2)] = 0.
        ite += 1

    # Final optimally extracted spectrum
    f_opt = np.sum(M_bp*P*D/V_new, axis=1) / (np.sum(M_bp*P*P/V_new, axis=1)+etol)
    # mask bad wavelength channels which contain more than 50% bad pixels
    badchannel = (np.sum(M_bp, axis=1) < len(spatial_x)*0.5)
    f_opt[badchannel] = np.nan
    # Rescale the variance by the chi2
    chi2_r = np.sum(Res)/(np.sum(M_bp)-len(f_opt))

    if chi2_r > 1:
        var = 1. / (np.sum(M_bp*P*P/V_new, axis=1)+etol) * chi2_r
    else:
        var = 1. / (np.sum(M_bp*P*P/V_new, axis=1)+etol)

    return f_opt, np.sqrt(var), [D.T, P.T], chi2_r


def wlen_solution(fluxes, errs, w_init, transm_spec=None, 
                p0_range=0.3, p1_range=0.03, p2_range=0.003,
                minimum_strength=0.005, debug=False):
    """
    Method for refining wavelength solution by maximizing 
    cross-correlation functions between the spectrum and a
    telluric transmission model on a order-by-order basis.

    Parameters
    ----------

    mode: str
        If mode is `linear`, then the wavelength solution is 
        corrected with a linear function. If mode is `quad`,
        the correction is a quadratic function.
    debug : bool
        generate plots for debugging.

    Returns
    -------
    wlens: array
        the refined wavelength solution
        
    """

    # Prepare a function to interpolate the skycalc transmission
    template_interp_func = interp1d(transm_spec[:,0], transm_spec[:,1], 
                                    kind='linear')

    def func_poly(poly, *args):
        wave, flux, err = args

        # Apply the polynomial coefficients
        new_wave = Poly.polyval(wave - np.mean(wave), poly) + wave

        # Interpolate the template onto the new wavelength grid
        template = template_interp_func(new_wave)

        # Minimize the chi-squared error, ignore the detector-edges
        chi_squared = np.nansum(((template - flux)[10:-10] / err[10:-10])**2)

        return chi_squared

    wlens = []
    for o in range(len(fluxes)):
        f, f_err, wlen_init = fluxes[o], errs[o], w_init[o]
        # normalize the sectrum
        nans = np.isnan(f)
        vmin, vmax = np.percentile(f[~nans], (1, 99)) 
        f, f_err = f/vmax, f_err/vmax

        index_o = (transm_spec[:,0]>np.min(wlen_init)) & \
                  (transm_spec[:,0]<np.max(wlen_init))

        # Check if there are enough telluric features in this wavelength range
        if np.std(transm_spec[:,1][index_o]) > minimum_strength: 
            
            # Use scipy.optimize to find the best-fitting coefficients
            res = optimize.minimize(
                        func_poly, 
                        args=(wlen_init[~nans], f[~nans], f_err[~nans]), 
                        x0=[0,0,0], method='Nelder-Mead', tol=1e-8, 
                        bounds=[(-p0_range,+p0_range),
                                (-p1_range,+p1_range), 
                                (-p2_range,+p2_range)]) 
            poly_opt = res.x
            if debug:
                print(poly_opt)

            # if the coefficient hits the prior edge, fitting is unsuccessful
            # fall back to the 0th oder solution.
            if np.isclose(np.abs(poly_opt[-1]), p2_range):
                res = optimize.minimize(
                        func_poly, 
                        args=(wlen_init[~nans], f[~nans], f_err[~nans]), 
                        x0=[0], method='Nelder-Mead', tol=1e-8, 
                        bounds=[(-p0_range,+p0_range),
                                # (-p1_range,+p1_range), 
                                ])
                poly_opt = res.x
                if debug:
                    print(poly_opt)
            wlen_cal = wlen_init + \
                    Poly.polyval(wlen_init - np.mean(wlen_init), poly_opt)     
        else:
            warnings.warn("Not enough telluric features to correct "
                         f"wavelength for order {o}")
            wlen_cal = wlen_init

        wlens.append(wlen_cal)

    return wlens


def wlen_solution_slow(fluxes, err, w_init, blazes, transm_spec=None, 
                    mode='quad', minimum_strength=0.005, debug=True):
    """
    Method for refining wavelength solution by maximizing 
    cross-correlation functions between the spectrum and a
    telluric transmission model on a order-by-order basis.

    Parameters
    ----------

    mode: str
        If mode is `linear`, then the wavelength solution is 
        corrected with a linear function. If mode is `quad`,
        the correction is a quadratic function.
    debug : bool
        generate plots for debugging.

    Returns
    -------
    wlens: array
        the refined wavelength solution
        
    """

    wlens = []
    for o in range(len(fluxes)):
        f, wlen_init, blaze = fluxes[o], w_init[o], blazes[o]
        f = f/blaze
        index_o = (transm_spec[:,0]>np.min(wlen_init)) & \
                  (transm_spec[:,0]<np.max(wlen_init))
        
        # Check if there are enough telluric features in this wavelength range
        if np.std(transm_spec[:,1][index_o]) > minimum_strength: 
            # Calculate the cross-correlation between data and template
            opt_poly = xcor_wlen_solution(f, wlen_init, transm_spec, mode=mode)

        else:
            warnings.warn("Not enough telluric features to correct "
                         f"wavelength for order {o}")
            opt_poly = [0., 1.0, 0.]

        result = [f'{item:.6f}' for item in opt_poly]

        mean_wavel = np.mean(wlen_init)
        if len(opt_poly) == 2:
            print(f"Order {o} -> Poly(x^0, x^1): {result}")
            wlen_cal = mean_wavel + opt_poly[0] \
                        + opt_poly[1] * (wlen_init - mean_wavel) 
        elif len(opt_poly) == 3:
            print(f"Order {o} -> Poly(x^0, x^1, x^2): {result}")
            wlen_cal = mean_wavel + opt_poly[0] \
                        + opt_poly[1] * (wlen_init - mean_wavel) \
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
        c_range: float = 0.001,
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
        N_b = int(N_a // 2 // 2 * 2 + 1)
        N_c = N_b
        # print(N_a, N_b)

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


def PolyfitClip(x, y, dg, m=None, w=None, clip=4., max_iter=10, \
                plotting=False):
    """
    Perform weighted least-square polynomial fit,
    iterratively cliping pixels above a certain sigma threshold
    
    Parameters
    ----------

    dg: int
        degree of polynomial 
    m: array
        if provided, it provides the mask.
    w: array 
        if provided, it provides the weights of each pixel.
    clip: int 
        sigma clip threshold
    max_iter: int 
        max number of iteration in sigma clip
    plotting: bool 
        if True, plot fitting and thresholds for debugging
    
    Returns
    ----------

    y_model: array
        polynomial fitted results
    poly: array
        Polynomial coefficience
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
