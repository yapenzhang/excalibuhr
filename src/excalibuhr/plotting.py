# File: src/excalibuhr/plotting.py
__all__ = []

# import os
# import sys
# import json
# import warnings
import numpy as np
import matplotlib.pyplot as plt 
from numpy.polynomial import polynomial as Poly

Nx = 2048
Ndet = 3
Norder = 7

def set_style():
    plt.rcParams.update({
        'font.size': 14,
        "xtick.labelsize": 14,   
        "ytick.labelsize": 14,   
        "xtick.direction": 'in', 
        "ytick.direction": 'in', 
        'ytick.right': True,
        'xtick.top': True,
        # "xtick.major.size": 5,
        # "xtick.minor.size": 2.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # "xtick.major.pad": 7,
        "lines.linewidth": 1,   
        'image.origin': 'lower',
        'image.cmap': 'viridis',
        "savefig.dpi": 300,   
    })

def plot_det_image(data, savename: str, title: str, tw = None, slit = None) -> None:
    # check data dimension
    data = np.array(data)
    if data.ndim == 3:
        Ndet = data.shape[0]
    elif data.ndim == 2:
        Ndet = 1
        data = data[np.newaxis,:]
    else:
        raise RuntimeError("Invalid data dimension") 

    xx = np.arange(data.shape[-1])

    set_style()
    fig, axes = plt.subplots(nrows=1, ncols=Ndet, 
                            figsize=(Ndet*4,4), constrained_layout=True)
    for i in range(Ndet):
        ax, im = axes[i], data[i]
        nans = np.isnan(im)
        vmin, vmax = np.percentile(im[~nans], (1, 99))
        ax.imshow(im, vmin=vmin, vmax=vmax)
        if not tw is None:
            trace = tw[i]
            trace_upper, trace_lower = trace
            for o, (poly_upper, poly_lower) in \
                    enumerate(zip(trace_upper, trace_lower)):
                yy_upper = Poly.polyval(xx, poly_upper)
                yy_lower = Poly.polyval(xx, poly_lower)
                ax.plot(xx, yy_upper, 'r')
                ax.plot(xx, yy_lower, 'r')
        if not slit is None:
            trace = tw[i]
            trace_upper, trace_lower = trace
            slit_meta = slit[i]
            meta0, meta1, meta2 = slit_meta
            for o, (poly_upper, poly_lower,  \
                poly_meta0, poly_meta1, poly_meta2) in enumerate(\
                zip(trace_upper,trace_lower, meta0, meta1, meta2)):
                yy_upper = Poly.polyval(xx, poly_upper)
                yy_lower = Poly.polyval(xx, poly_lower)
                poly_full = np.array([Poly.polyval(xx, poly_meta0), 
                                  Poly.polyval(xx, poly_meta1),
                                  Poly.polyval(xx, poly_meta2)]).T
                yy = np.arange(int(yy_lower.min()-1), int(yy_upper.max()+1))
                for x in np.arange(0, len(xx), 100)[1:-1]:
                    ax.plot(Poly.polyval(yy, poly_full[x]), yy, ':r')

    plt.suptitle(title, y=0.98)
    plt.savefig(savename[:-4]+'png')
    plt.close(fig)



def plot_spec1d(wlen, flux, err, savename):
    set_style()
    fig, axes = plt.subplots(nrows=Norder, ncols=1, 
                    figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        nans = np.isnan(flux[indices])
        vmin, vmax = np.percentile(flux[indices][~nans], (1, 99))
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], 
                         flux[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.6*vmin, 1.2*vmax))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Flux')
    plt.savefig(savename[:-4]+'.png')
    plt.close(fig)

    fig, axes = plt.subplots(nrows=Norder, ncols=1, 
                    figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        nans = np.isnan(flux[indices])
        vmin, vmax = np.percentile((flux/err)[indices][~nans], (1, 99))
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], 
                        (flux/err)[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.6*vmin, 1.2*vmax))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Flux')
    plt.savefig(savename[:-4]+'_SNR.png')
    plt.close(fig)