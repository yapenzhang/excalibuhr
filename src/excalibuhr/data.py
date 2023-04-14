
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy import ndimage
import excalibuhr.utils as su 
import copy 

class SPEC2D:
    """
    Object for spectral data in 2D shape (N_chip x N_pixel)
    """

    def __init__(self, wlen=None, flux=None, err=None,
                 filename=None, fmt="ext3") -> None:
        """
        initilize the object either with arrays passed via variables, 
        `wlen`, `flux`, and `err`, or with data read from text files 
        and FITS files. 
        
        fmt: string
            

        """
        
        if filename is None and wlen is not None:
            self.wlen = wlen
            self.flux = flux
            self.err = err
            self.reformat_data()
        else:
            self.wlen, self.flux, self.err = \
                    np.genfromtxt(filename, skip_header=1, unpack=True)
            self.reformat_data()


    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        self._header = value

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


    def make_covariance(self, amp, length, trunc_dist=4):

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

    
    def match_LSF(self, wlen_sharp, flux_sharp, chip_bin=3, kernel_size=50):
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
            # interp1d(spec_denominator.wlen, spec_denominator.flux, 
            #          bounds_error=False, 
            #                   fill_value=np.nan)
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
        Nchips = self.wlen.shape[0]
        spec = self._copy()
        for i in range(Nchips):
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

    def high_pass_filter(self, sigma=201):
        Nchips = self.wlen.shape[0]
        spec = self._copy()
        for i in range(Nchips):
            spec.flux[i] /= ndimage.gaussian_filter(self.flux[i], sigma=sigma)
            spec.err[i] /= ndimage.gaussian_filter(self.flux[i], sigma=sigma)
        return spec


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
        Nchips = self.wlen.shape[0]
        self.wlen = su.wlen_solution(self.flux, self.err, self.wlen, transm_spec, debug=debug)
        if debug:
            for i in range(Nchips):
                plt.plot(self.wlen[i], self.flux[i], 'k', alpha=0.8)
            plt.plot(transm_spec[:,0], transm_spec[:,1], 'b', alpha=0.8)
            plt.show()
        return self

    def _set_plot_style(self):
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
        for dt in [self.wlen, self.flux, self.err]:
            unraveled.append(dt.flatten())
        wlen, spec, err = unraveled
        header = "Wlen(nm) Flux Flux_err"
        np.savetxt(savename, np.c_[wlen, spec, err], header=header)


    def plot_spec1d(self, savename, show=False):
        self._set_plot_style()
        nrows = self.wlen.shape[0]//3
        if self.wlen.shape[0]%3 != 0:
            nrows += 1
        fig, axes = plt.subplots(nrows=nrows, ncols=1, 
                        figsize=(12,nrows), constrained_layout=True)
        for i in range(nrows):
            ax = axes[i]
            wmin, wmax = self.wlen[i*3][0], self.wlen[min(i*3+2, self.wlen.shape[0]-1)][-1]
            ymin, ymax = 1, 0
            for j in range(min(3, self.wlen.shape[0]-3*i)):
                x, y = self.wlen[i*3+j], self.flux[i*3+j]
                ax.plot(x, y, 'k')
                nans = np.isnan(y)
                vmin, vmax = np.percentile(y[~nans], (1, 99))
                ymin = min(vmin, ymin)
                ymax = max(vmax, ymax)
            ax.set_xlim((wmin, wmax))
            ax.set_ylim((vmin, vmax))
        axes[-1].set_xlabel('Wavelength (nm)')
        axes[-1].set_ylabel('Flux')
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
                    nans = np.isnan(y)
                    vmin, vmax = np.percentile((y/z)[~nans], (10, 90))
                    ymin = min(vmin, ymin)
                    ymax = max(vmax, ymax)
                ax.set_xlim((wmin, wmax))
                ax.set_ylim((0.4*vmin, 1.3*vmax))
            axes[-1].set_xlabel('Wavelength (nm)')
            axes[-1].set_ylabel('S/N')
            plt.savefig(savename[:-4]+'_SNR.png')
            plt.close(fig)


