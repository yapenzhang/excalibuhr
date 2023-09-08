# File: src/excalibuhr/data.py
__all__ = ['SPEC2D']

import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy import stats
from scipy.interpolate import interp1d
from scipy import ndimage, signal
import excalibuhr.utils as su 
import copy 

class SPEC2D:
    """
    Object for spectral data in 2D shape (N_chip x N_pixel)
    It contains the wavelength, flux, and error arrays. 
    It has several methods for manipulating and analyzing the data.
    """

    def __init__(self, wlen=None, flux=None, err=None,
                 filename=None, fmt="ext3") -> None:
        """
        initilize the object either with arrays passed via variables, 
        `wlen`, `flux`, and `err`, or with data read from text files 
        and FITS files. 
        
        fmt: string
            
        """
        
        if filename is not None:
            ext = filename.split('.')[-1]
            if ext == "fits":
                if fmt == "ext3":
                    with fits.open(filename) as hdu:
                        self.header = hdu[0].header
                        self.flux = hdu['FLUX'].data
                        self.err = hdu['FLUX_ERR'].data
                        self.wlen = hdu['WAVE'].data
                elif fmt == "molecfit":
                    w, f, f_err = [], [], []
                    with fits.open(filename) as hdu:
                        self.header = hdu[0].header
                        for i in range(1, len(hdu)):
                            w.append(hdu[i].data['WAVE'])
                            f.append(hdu[i].data['FLUX'])
                            f_err.append(hdu[i].data['FLUX_ERR'])
                    self.flux = np.array(f)
                    self.err = np.array(f_err)
                    self.wlen = np.array(w)
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

    def wlen_cut(self, w_list):
        # w_list: list of wavelength ranges to keep.
        w_list = np.array(w_list)
        if w_list.ndim > 1:
            cmin = w_list[:,0]
            indices = np.argsort(cmin)
            w_list = w_list[indices]
        else:
            w_list = w_list[np.newaxis,:]
        mask = np.zeros_like(self.wlen, dtype=bool)
        for cut in w_list:
            mask = mask | ((self.wlen>cut[0]) & (self.wlen<cut[1]))
        spec = self._copy()
        spec.flux[~mask] = np.nan
        chip_mask = np.sum(np.isnan(spec.flux), axis=1) < 0.9 * self.wlen.shape[-1]
        # remove fully masked orders
        return self._copy(self.wlen[chip_mask], self.flux[chip_mask], 
                          self.err[chip_mask])


    def remove_order_edge(self, Nedge=10):
        return self._copy(self.wlen[:, Nedge:-Nedge], self.flux[:, Nedge:-Nedge],
                          self.err[:, Nedge:-Nedge])

    def make_wlen_bins(self):
        data_wlen_bins = np.zeros_like(self.wlen)
        for i in range(self.Nchip):
            data_wlen_bins[i][:-1] = np.diff(self.wlen[i])
            data_wlen_bins[i][-1] = data_wlen_bins[i][-2]
        self.wlen_bins = data_wlen_bins


    def make_covariance(self, amp, length, trunc_dist=5):

        if np.array(amp).size == 1: 
            amp = np.ones(self.Nchip) * amp[0]
            length = np.ones(self.Nchip) * length[0]
        elif np.array(amp).size != self.Nchip:
            raise Exception("GP kernel parameters should have the same size as the number of data chips")
        
        self.cov = []
        delta_wave = np.abs(self.wlen[:,:,None] - self.wlen[:,None,:])

        for i in range(self.Nchip):
            cov = su.add_RBF_kernel(amp[i], length[i], delta_wave[i], 
                                      self.err[i], trunc_dist=trunc_dist)
            self.cov.append(cov)
            

    def make_covariance_local(self, amp, mu, sigma, trunc_dist=5):
        
        for a, m, s in zip(amp, mu, sigma):
            # check which chip the local feature belongs to
            indices = np.searchsorted(self.wlen[:,0], m)-1
            cov_local = su.add_local_kernel(a, m, s, self.wlen[indices], trunc_dist=trunc_dist)
            self.cov[indices] += cov_local
    
    def make_spline_model(self, N_knots):
        x_knots = np.array([np.linspace(self.wlen[i][0], self.wlen[i][-1], N_knots) for i in range(self.Nchip)])
        M_spline = su.get_spline_model(x_knots, self.wlen)
        splined = M_spline * self.flux[:, :, None]
        return splined, M_spline
    
    def match_LSF(self, wlen_sharp, flux_sharp, chip_bin=3, kernel_size=20):
        spec_reconst = []
        spec_sharp = interp1d(wlen_sharp, flux_sharp, bounds_error=False, 
                              fill_value=np.nanmean(flux_sharp))
        for i in range(0, self.Nchip, chip_bin):
            xx = self.wlen[i:i+chip_bin].flatten()
            yy = self.flux[i:i+chip_bin].flatten()
            flux_reconst, Kernel = su.find_kernel_SVD(
                                        spec_sharp(xx), 
                                        yy, 
                                        kernel_size)
            spec_reconst.append(flux_reconst)
        return self._copy(self.wlen.flatten(), spec_reconst)


    def spec_division(self, spec_denominator):
        """
        spec_denominator: SPEC2D object with the same wlen as the data
        """

        if not np.allclose(self.wlen, spec_denominator.wlen):
            pass
        else:
            return self._copy(flux=self.flux/spec_denominator.flux, 
                             err=self.err/spec_denominator.flux)


    def remove_blackbody(self, teff, debug=False):
        from astropy.modeling import models
        from astropy import units as u
        bb = models.BlackBody(temperature=teff * u.K,
                 scale=1.0 * u.watt / (u.m ** 2 * u.micron * u.sr))
        norm = []
        for i in range(self.Nchip):
            flux_lambda = bb(self.wlen[i] * u.nm) * np.pi * u.steradian
            norm.append(flux_lambda.value)
        norm = np.array(norm)/np.median(norm)
        if debug:
            for i in range(self.Nchip):
                plt.plot(self.wlen[i], self.flux[i], 'k')
                plt.plot(self.wlen[i], self.flux[i]/norm[i],'r')
            plt.show()
        return self._copy(flux=self.flux/norm, err=self.err/norm)


    def continuum_normalization(self, order=3, debug=False):
        spec = self._copy()
        for i in range(self.Nchip):
            if order == 0:
                nans = np.isnan(self.flux[i])
                _, continuum = np.percentile(self.flux[i][~nans], (1, 99))
            else:
                continuum, _ = su.fit_continuum(self.wlen[i], self.flux[i], order)
            spec.flux[i] /= continuum
            spec.err[i] /= continuum
            if debug:
                plt.plot(self.wlen[i], self.flux[i], 'k')
                plt.plot(self.wlen[i], continuum, 'r')
        if debug:
            plt.show()
        return spec

    def high_pass_filter(self, sigma=51):
        #or signal.savgol_filter
        spec = self._copy()
        for i in range(self.Nchip):
            spec.flux[i] /= ndimage.gaussian_filter(self.flux[i], sigma=sigma)
            spec.err[i] /= ndimage.gaussian_filter(self.flux[i], sigma=sigma)
            # spec.flux[i] -= np.nanmean(spec.flux[i])
            # plt.plot(self.flux[i])
            # plt.plot(ndimage.gaussian_filter(self.flux[i], sigma=sigma))
            # plt.show()
        return spec

    def get_outlier_mask(self, clip=3):
        outlier_mask = []
        for i in range(self.Nchip):
            filtered = stats.sigma_clip(self.flux[i], sigma=clip)
            outlier_mask.append(filtered.mask | np.isnan(self.flux[i]))
        self.mask = outlier_mask

    # def remove_nans(self):
    #     spec = self._copy()
    #     w, f, f_err = [], [], []
    #     for i in range(self.Nchip):
    #         nans = np.isnan(self.flux[i])
    #         w.append(self.wlen[i][~nans])
    #         f.append(self.flux[i][~nans])
    #         f_err.append(self.err[i][~nans])
    #     spec.wlen = w
    #     spec.flux = f
    #     spec.err = f_er
    #     return spec
    
    # def outliers(self, spec_model, clip=3):
    #     res = self._copy(flux=self.flux-spec_model.flux)
    #     for i in range(self.Nchip):
    #         filtered = stats.sigma_clip(res.flux[i], sigma=clip)
    #         outlier_mask = filtered.mask
    #         plt.plot(self.wlen[i], res.flux[i], 'k')
    #         for w in self.wlen[i][outlier_mask]:
    #             plt.axvline(w, ls=':', color='r', alpha=0.7)
    #     plt.show()


    def noise_stat(self, spec_model, Nbins=20):
        res = self._copy(flux=self.flux-spec_model.flux)
        res = res.high_pass_filter()
        x, y, _ = self.get_spec1d()
        x, r, _ = res.get_spec1d()
        r = np.abs(r/y)
        ybins = np.linspace(np.min(y), np.max(y), Nbins)
        ybinned = (ybins[1:]+ybins[:-1])/2.
        indices = np.digitize(y, ybins)
        rbinned = np.array([np.mean(r[indices==i]) for i in range(1, Nbins)])
        rbinned_err = np.array([np.std(r[indices==i])/np.sqrt(np.sum(indices==i)) 
                                for i in range(1, Nbins)])
        return ybinned, rbinned, rbinned_err


    def doppler_shift_dw(self, dw):
        return self._copy(wlen=self.wlen+dw)

    def wlen_calibration(self, transm_spec, debug=False):
        self.wlen = su.wlen_solution(self.flux, self.err, self.wlen, transm_spec, debug=debug)
        if debug:
            for i in range(self.Nchip):
                plt.plot(self.wlen[i], self.flux[i], 'k', alpha=0.8)
            plt.plot(transm_spec[:,0], transm_spec[:,1], 'b', alpha=0.8)
            plt.show()
        return self

    def _set_plot_style(self):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({
            'font.size': 10,
            "xtick.labelsize": 10,   
            "ytick.labelsize": 10,   
            "xtick.direction": 'in', 
            "ytick.direction": 'in', 
            'ytick.right': True,
            'xtick.top': True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # "xtick.major.size": 5,
            # "xtick.minor.size": 2.5,
            "lines.linewidth": 0.5,   
            'image.origin': 'lower',
            'image.cmap': 'cividis',
            "savefig.dpi": 300,   
            })


    def save_spec1d(self, savename):
        unraveled = []
        if self.err is None:
            arr = [self.wlen, self.flux]
        else:
            arr = [self.wlen, self.flux, self.err]
        for dt in arr:
            unraveled.append(dt.flatten())
        header = "Wlen(nm) Flux Flux_err"
        np.savetxt(savename, np.transpose(unraveled), header=header)


    def plot_spec1d(self, savename, show=False):
        self._set_plot_style()
        nrows = self.wlen.shape[0]//3
        if self.wlen.shape[0]%3 != 0:
            nrows += 1
        fig, axes = plt.subplots(nrows=nrows, ncols=1, 
                        figsize=(12,nrows), constrained_layout=True)
        if nrows == 1:
            axes = [axes]
        for i in range(nrows):
            ax = axes[i]
            wmin, wmax = self.wlen[i*3][0], self.wlen[min(i*3+2, self.wlen.shape[0]-1)][-1]
            ymin, ymax = 1e8, 0
            for j in range(min(3, self.wlen.shape[0]-3*i)):
                x, y = self.wlen[i*3+j], self.flux[i*3+j]
                ax.plot(x, y, 'k')
                nans = np.isnan(y)
                vmin, vmax = np.percentile(y[~nans], (1, 99))
                ymin = min(vmin, ymin)
                ymax = max(vmax, ymax)
            ax.set_xlim((wmin, wmax))
            ax.set_ylim((0.8*vmin, 1.2*vmax))
        axes[-1].set_xlabel('Wavelength (nm)')
        axes[nrows//2].set_ylabel('Flux')
        plt.savefig(savename)
        if show:
            plt.show()
        plt.close(fig)

        if self.err is not None:
            fig, axes = plt.subplots(nrows=nrows, ncols=1, 
                        figsize=(12,nrows), constrained_layout=True)
            for i in range(nrows):
                ax = axes[i]
                # wmin, wmax = self.wlen[i*3][0], self.wlen[i*3+2][-1]
                wmin, wmax = self.wlen[i*3][0], self.wlen[min(i*3+2, self.wlen.shape[0]-1)][-1]
                ymin, ymax = 1, 0
                for j in range(min(3, self.wlen.shape[0]-3*i)):
                    x, y, z = self.wlen[i*3+j], self.flux[i*3+j], self.err[i*3+j]
                    ax.plot(x, y/z, 'k')
                    nans = np.isnan(y/z)
                    vmin, vmax = np.percentile((y/z)[~nans], (10, 90))
                    ymin = min(vmin, ymin)
                    ymax = max(vmax, ymax)
                ax.set_xlim((wmin, wmax))
                ax.set_ylim((0.4*vmin, 1.3*vmax))
            axes[-1].set_xlabel('Wavelength (nm)')
            axes[-1].set_ylabel('S/N')
            plt.savefig(savename[:-4]+'_SNR.png')
            plt.close(fig)
