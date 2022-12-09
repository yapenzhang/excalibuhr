# File: src/excalibuhr/plotting.py
__all__ = [
    "plot_det_image", 
    "plot_spec1d",
    "plot_extr_model",
    ]

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

def plot_det_image(data, savename: str, title: str, tw = None, slit = None, x_fpets = None) -> None:
    # check data dimension
    data = np.array(data)
    if data.ndim == 3:
        Ndet = data.shape[0]
    elif data.ndim == 2:
        Ndet = 1
        data = data[np.newaxis,:]
    else:
        raise TypeError("Invalid data dimension") 

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

                xx_lines = x_fpets[i][o]
                yy_upper = Poly.polyval(xx_lines, poly_upper)
                yy_lower = Poly.polyval(xx_lines, poly_lower)
                poly_full = np.array([Poly.polyval(xx_lines, poly_meta0), 
                                  Poly.polyval(xx_lines, poly_meta1),
                                  Poly.polyval(xx_lines, poly_meta2)]).T
                yy = np.arange(int(yy_lower.min()-1), int(yy_upper.max()+1))
                for x in range(len(xx_lines)):
                    ax.plot(Poly.polyval(yy, poly_full[x]), yy, ':r', zorder=10)

    plt.suptitle(title, y=0.98)
    plt.savefig(savename[:-4]+'png')
    plt.close(fig)

def plot_extr_model(D, chi2, savename):
    set_style()
    # check data dimension
    D = np.array(D)
    Ndet, Norder = D.shape[0], D.shape[1]
    fig, axes = plt.subplots(nrows=Norder*2, ncols=Ndet, 
                    figsize=(2*Norder, 14), sharex=True, sharey=True,  
                    constrained_layout=True)
    for i in range(Ndet):
        for o in range(0, 2*Norder, 2):
            ax_d, ax_m = axes[Norder*2-o-2, i], axes[Norder*2-o-1, i] 
            data, model = D[i, o//2, 0], D[i, o//2, 1]
            nans = np.isnan(data)
            vmin, vmax = np.percentile(data[~nans], (5, 95))
            ax_d.imshow(data, vmin=vmin, vmax=vmax, aspect='auto')
            ax_m.imshow(model, vmin=0, vmax=np.max(model), aspect='auto')
            ax_m.set_title(r"Order {0}, $\chi_r^2$: {1:.2f}".format(o//2, chi2[i, o//2]))
        ax_d.set_title(f"Detector {i}")
    plt.savefig(savename[:-4]+'png')
    plt.close(fig)


def plot_spec1d(wlen, flux, err, savename):
    set_style()
    fig, axes = plt.subplots(nrows=Norder, ncols=1, 
                    figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        nans = np.isnan(flux[indices])
        vmin, vmax = np.percentile(flux[indices][~nans], (10, 90))
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], 
                         flux[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.4*vmin, 1.3*vmax))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Flux')
    plt.savefig(savename[:-4]+'.png')
    plt.close(fig)

    fig, axes = plt.subplots(nrows=Norder, ncols=1, 
                    figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        nans = np.isnan(flux[indices])
        vmin, vmax = np.percentile((flux/err)[indices][~nans], (10, 90))
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], 
                        (flux/err)[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.4*vmin, 1.3*vmax))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('S/N')
    plt.savefig(savename[:-4]+'_SNR.png')
    plt.close(fig)