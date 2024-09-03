# File: src/excalibuhr/data.py
__all__ = ['SPEC', 'SERIES', 'DETECTOR']

import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy import stats
from scipy.interpolate import interp1d
from scipy import ndimage, signal
import excalibuhr.utils as su 
import copy 


class SERIES:
    """
    Object for spectral data in 2D shape (N_chip x N_pixel)
    It contains the wavelength, flux, and error arrays. 
    It has several methods for manipulating and analyzing the data.
    """

    def __init__(self, filename=None, wlen=None, flux=None, err=None, header=None):
        if filename is not None:
            ext = filename.split('.')[-1]
            if ext == "fits":
                with fits.open(filename) as hdu:
                    self.header = hdu[0].header
                    self.flux = hdu['FLUX'].data
                    self.err = hdu['FLUX_ERR'].data
                    self.wlen = hdu['WAVE'].data
        elif wlen is not None:
            self.wlen = wlen
            self.flux = flux
            self.err = err
            self.header = header
        
        self.data = [SPEC(wlen=self.wlen, flux=self.flux[i], 
                          err=self.err[i], header=self.header
                          ) for i in range(len(self.flux))]

    def __iter__(self):
        return self.data.__iter__()
    

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self.data[indices]
        else:
            return SERIES(specs=[self.data[ind] for ind in indices])


    def __len__(self):
        return len(self.data)


class SPEC:
    """
    Object for spectral data in 2D shape (N_chip x N_pixel)
    It contains the wavelength, flux, and error arrays. 
    It has several methods for manipulating and analyzing the data.
    """

    def __init__(self, filename=None, wlen=None, flux=None, err=None, header=None):
        """
        initilize the object either with arrays passed via variables, 
        `wlen`, `flux`, and `err`, or with data read from text files 
        and FITS files. 
        
        fmt: string
            
        """
        
        if filename is not None:
            ext = filename.split('.')[-1]
            if ext == "fits":
                # if fmt == "ext3":
                with fits.open(filename) as hdu:
                    self.header = hdu[0].header
                    self.flux = hdu['FLUX'].data
                    self.err = hdu['FLUX_ERR'].data
                    self.wlen = hdu['WAVE'].data
                
                # elif fmt == "molecfit":
                #     w, f, f_err = [], [], []
                #     with fits.open(filename) as hdu:
                #         self.header = hdu[0].header
                #         for i in range(1, len(hdu)):
                #             w.append(hdu[i].data['WAVE'])
                #             f.append(hdu[i].data['FLUX'])
                #             f_err.append(hdu[i].data['FLUX_ERR'])
                #     self.flux = np.array(f)
                #     self.err = np.array(f_err)
                #     self.wlen = np.array(w)
            else:
                try:
                    self.wlen, self.flux, self.err = \
                        np.genfromtxt(filename, skip_header=1, unpack=True)
                except:
                    self.wlen, self.flux = \
                        np.genfromtxt(filename, skip_header=1, unpack=True)
                    self.err = None
            self.reformat_data()
        elif wlen is not None:
            self.wlen = wlen
            self.flux = flux
            self.err = err
            self.header = header
            self.reformat_data()


    def reformat_data(self):
        if not isinstance(self.wlen, np.ndarray):
            self.wlen = np.array(self.wlen)
            self.flux = np.array(self.flux)
            if self.err is not None:
                self.err = np.array(self.err)

        if self.wlen.ndim == 1:
            # detect gaps in the wavelength, obtain the number of chuncks
            diffs = np.diff(self.wlen)
            ind_edge = np.argwhere(diffs/np.median(diffs)>20).flatten()
            if ind_edge.size > 0:
                ind_edge = np.insert(ind_edge+1, 0, 0)
                ind_edge = np.insert(ind_edge, len(ind_edge), len(self.wlen))
                Nchip = len(ind_edge)-1
            else:
                Nchip = 1
            self.wlen = np.reshape(self.wlen, (Nchip, -1)) 
            self.flux = np.reshape(self.flux, (Nchip, -1))
            if self.err is not None:
                self.err = np.reshape(self.err, (Nchip, -1)) 
        elif self.wlen.ndim == 3:
            # detector x order x spec
            self.wlen.reshape((-1, self.wlen.shape[-1]))
            self.flux.reshape((-1, self.flux.shape[-1]))
            if self.err is not None:
                self.err.reshape((-1, self.err.shape[-1]))

        wmin = self.wlen[:,0] 
        indices = np.argsort(wmin)
        self.wlen = self.wlen[indices]
        self.flux = self.flux[indices]
        if self.err is not None:
            self.err = self.err[indices]
        self.Nchip = self.wlen.shape[0]


    def _copy(self, wlen=None, flux=None, err=None):
        dt = copy.deepcopy(self)
        if wlen is not None:
            dt.wlen = wlen 
        if flux is not None:
            dt.flux = flux 
        if err is not None:
            dt.err = err
        dt.reformat_data()
        return dt

    def get_spec1d(self):
        if self.err is not None:
            return self.wlen.flatten(), self.flux.flatten(), self.err.flatten()
        else:
            return self.wlen.flatten(), self.flux.flatten(), None

class DETECTOR:
    """
    Object for a list of detectors data in 2D shape (N_spatial_pixel x N_spectral_pixel)

    """

    def __init__(self, data=None, fields=None, filename=None):
        """
        initilize the object either with arrays passed via `data`. 
        fields of data may contain flux, variance, psf, etc.

        """
        if data is not None:
            for i, key in enumerate(fields):
                setattr(self, key, data[i])
            self.Ndet = len(data[0])
            self.Norder = len(data[0][0])
        elif filename is not None:
            self.load_extr2d(filename)


    def save_extr2d(self, filename):
        """
        Method for saving the pipeline extracted 2D spectral data to .npz files.
        The 2D data shape can vary in the spatial diemsion across different orders, 
        therefore cannot be directly stored as numpy ndarray.
        The data of all orders are stacked along the 0th axis and then saved as numpy arrays.   

        Parameters
        ----------
        filename : str
            Path to save the 2d data as a `.npz` file. A list of 2D data 
            (such as flux, variance, fitted spatial profile) 
            will be be stacked and saved.

        Returns
        -------
        NoneType
            None
        """
        all_stacked = {}
        for key in self.__dict__.keys() - ['Ndet', 'Norder']:
            dt = self.__getattribute__(key)
            dt_stack, id_orders = [], []
            for dt_per_det in dt:
                order_stack, id_order = stack_ragged(dt_per_det)
                dt_stack.append(order_stack)
                id_orders.append(id_order)
            dt_stack, id_dets = stack_ragged(dt_stack)
            all_stacked[key] = dt_stack
        all_stacked['id_dets'] = id_dets
        all_stacked['id_orders'] = id_orders
        np.savez(filename, **all_stacked)


    def load_extr2d(self, filename):
        """
        Method for reading and unraveling the pipeline 2D extracted .npz files into arrays.

        Parameters
        ----------
        filename : str
            Path of the `EXTR2D` .npz file to load. The 2D data will be unraveled to the 4d 
            shape of (N_detector, N_order, N_spatial_pixel, N_dispersion_pixel), and set as
            the attributes of the class.

        Returns
        -------
        NoneType
            None
        """

        data = np.load(filename)
        id_dets = data['id_dets']
        id_orders = data['id_orders']
        self.Ndet, self.Norder = id_dets.shape[0]+1, id_orders.shape[1]+1
        for key in data.keys() - ['id_dets', 'id_orders']:
            dt = data[key]
            D_unravel = []
            D_det = np.split(dt, id_dets)
            for i in range(self.Ndet):
                D_order = np.split(D_det[i], id_orders[i])
                D_unravel.append(D_order)
            setattr(self, key, D_unravel)
    
    
    def plot_extr2d_model(self, savename):
        _set_plot_style()

        fig, axes = plt.subplots(nrows=self.Norder*2, ncols=self.Ndet, 
                        figsize=(2*self.Norder, 14), sharex=True, sharey=True,  
                        constrained_layout=True)
        for i in range(self.Ndet):
            D_order = self.flux[i]
            P_order = self.psf[i]
            rows_crop = 0
            for o in range(0, 2*self.Norder, 2):
                ax_d, ax_m = axes[self.Norder*2-o-2, i], axes[self.Norder*2-o-1, i] 
                data, model = D_order[o//2], P_order[o//2]
                rows_crop = max(data.shape[0], rows_crop)
                if data.size != 0:
                    nans = np.isnan(data)
                    vmin, vmax = np.percentile(data[~nans], (5, 95))
                    ax_d.imshow(data, vmin=vmin, vmax=vmax, aspect='auto')
                    ax_m.imshow(model, vmin=0, vmax=np.max(model), aspect='auto')
                ax_d.set_title(f"Order {o//2}")
            axes[-1,i].set_xlabel(f"Detector {i}", size='large', fontweight='bold')
        axes[0,0].set_ylim((0, rows_crop))
        plt.savefig(savename+'.png')
        plt.close(fig)


def wfits(fname, ext_list: dict, header=None):
    """
    write data to FITS primary and extensions, overwriting any old file
    
    Parameters
    ----------
    fname: str
        path and filename to which the data is saved 
    ext_list: dict
        to save the data in the dictionary to FITS extension. Specify the datatype 
        (e.g.  "FLUX", "FLUX_ERR", "WAVE", and "MODEL") in the key. 
    header: FITS `header`
        header information to be saved

    Returns
    -------
    NoneType
        None
    """

    primary_hdu = fits.PrimaryHDU(header=header)
    new_hdul = fits.HDUList([primary_hdu])
    if not ext_list is None:
        for key, value in ext_list.items():
            new_hdul.append(fits.ImageHDU(value, name=key))
    new_hdul.writeto(fname, overwrite=True, output_verify='ignore') 


def stack_ragged(array_list, axis=0):
    """
    stack arrays with same number of columns but different number of rows
    into a new array and record the indices of each sub array.

    Parameters
    ----------
    array_list: list
        the list of array to be stacked

    Returns
    -------
    stacked: array
        the stacked array along the 0th axis
    idx: array
        the indices to retrieve back the sub arrays
    """
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx


def _set_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        "xtick.labelsize": 12,   
        "ytick.labelsize": 12,   
        "xtick.direction": 'in', 
        "ytick.direction": 'in', 
        'ytick.right': True,
        'xtick.top': True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # "xtick.major.size": 5,
        # "xtick.minor.size": 2.5,
        # "xtick.major.pad": 7,
        "lines.linewidth": 0.5,   
        'image.origin': 'lower',
        'image.cmap': 'cividis',
        "savefig.dpi": 300,   
        })