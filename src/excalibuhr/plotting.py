# File: src/excalibuhr/plotting.py
__all__ = []

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from astropy.io import fits

plt.rcParams.update({
    # "axes.facecolor": "black",
    'font.size': 13,
    'lines.linewidth': 2,
    'ytick.right': True,
    'xtick.top': True,
    'image.origin': 'lower',
    'image.cmap': 'viridis',
            })
# import matplotlib as mpl
# mpl.rc('xtick', labelsize=13, direction='in', color='white', labelcolor='k')
# mpl.rc('ytick', labelsize=13, direction='in', color='white', labelcolor='k')
# mpl.rc('xtick.major', size=6, width=1)
# mpl.rc('ytick.major', size=6, width=1)
# mpl.rc('xtick.minor', size=3, width=0.5, visible=True)
# mpl.rc('ytick.minor', size=3, width=0.5, visible=True)

# def plot_det_image(filename, ):
