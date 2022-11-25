# File: src/excalibuhr/plotting.py
__all__ = []

# import os
# import sys
# import json
# import warnings
import numpy as np
import matplotlib.pyplot as plt 

Nx = 2048
Ndet = 3
Norder = 7

def set_style():
    plt.rcParams.update({
        'font.size': 15,
        "xtick.labelsize": 15,   
        "ytick.labelsize": 15,   
        "xtick.direction": 'in', 
        "ytick.direction": 'in', 
        'ytick.right': True,
        'xtick.top': True,
        # "xtick.major.size": 5,
        # "xtick.minor.size": 2.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # "xtick.major.pad": 7,
        "lines.linewidth": 2,   
        'image.origin': 'lower',
        'image.cmap': 'viridis',
        "savefig.dpi": 200,   
    })

def plot_spec1d(wlen, flux, err, savename):
    set_style()
    fig, axes = plt.subplots(nrows=Norder, ncols=1, figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        mask = np.isnan(flux[indices])
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], flux[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.6*np.median(np.sort(flux[indices][~mask])[:len(indices)//20]), \
                        1.2*np.median(np.sort(flux[indices][~mask])[-len(indices)//20:])))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Flux')
    plt.savefig(savename+'.png')
    plt.close(fig)

    fig, axes = plt.subplots(nrows=Norder, ncols=1, figsize=(14,2*Norder), constrained_layout=True)
    for i in range(Norder):
        indices = range(i*Ndet*Nx, (i+1)*Ndet*Nx)
        mask = np.isnan(flux[indices])
        for d in range(Ndet):
            indices_det = range(d*Nx, (d+1)*Nx)
            axes[i].plot(wlen[indices][indices_det], (flux/err)[indices][indices_det], 'k')
        axes[i].set_xlim((wlen[indices][0], wlen[indices][-1]))
        axes[i].set_ylim((0.6*np.median(np.sort((flux/err)[indices][~mask])[:len(indices)//20]), \
                        1.2*np.median(np.sort((flux/err)[indices][~mask])[-len(indices)//20:])))
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Flux')
    plt.savefig(savename+'_SNR.png')
    plt.close(fig)