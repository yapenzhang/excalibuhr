
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from scipy import ndimage
import excalibuhr.utils as su 
import copy 

class SPEC2D:
    """
    Object for spectral data in 2D shape (N_chip x N_pixel)
    """

    def __init__(self, filename=None, fmt="ext3", 
                 wlen=None, flux=None, err=None, Npix=2048) -> None:
        """
        initilize the object either with arrays passed via variables, 
        `wlen`, `flux`, and `err`, or with data read from text files 
        and FITS files. 
        
        fmt: string
            

        """
        self.Npix = Npix
        
        if filename is None:
            self.wlen = wlen
            self.flux = flux
            self.err = err
        else:
            ext = filename.split('.')[-1]
            if ext == "fits":
                if fmt == "ext3":
                    with fits.open(filename) as hdu:
                        self.header = hdu[0].header
                        self.flux = hdu[0].data
                        self.err = hdu[1].data
                        self.wlen = hdu[2].data
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
                    raise NotImplementedError("Please read in the fits data manually.")
            else:
                self.wlen, self.flux, self.err = \
                        np.genfromtxt(filename, skip_header=1, unpack=True)

        if self.wlen is not None:
            if not isinstance(self.wlen, np.ndarray):
                self.wlen = np.array(self.wlen)
                self.flux = np.array(self.flux)
                if self.err is not None:
                    self.err = np.array(self.err)
            if self.wlen.ndim == 1:
                # self.wlen = self.wlen[np.newaxis,:]
                # self.flux = self.flux[np.newaxis,:]
                self.wlen = np.reshape(self.wlen, (-1, self.Npix)) 
                self.flux = np.reshape(self.flux, (-1, self.Npix))
                if self.err is not None:
                    self.err = np.reshape(self.err, (-1, self.Npix)) 
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


    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        self._header = value

    def _copy(self, wlen=None, flux=None, err=None, Npix=None):
        if wlen is None:
            wlen = self.wlen 
        if flux is None:
            flux = self.flux 
        if err is None:
            err = self.err
        if Npix is None:
            Npix = self.Npix
        dt = SPEC2D(wlen=wlen, flux=flux, err=err, Npix=Npix)
        if hasattr(self, "header"):
            dt.header = self.header
        return dt
    # def _copy(self):
    #     return copy.deepcopy(self)

    def get_spec1d(self):
        if self.err is not None:
            return self.wlen.flatten(), self.flux.flatten(), self.err.flatten()
        else:
            return self.wlen.flatten(), self.flux.flatten(), None

    def wlen_cut(self, w0, w1):
        # list of wavelength ranges to keep, 
        # w0: the lower bounds, w1: the upper bounds.
        mask = np.zeros_like(self.wlen, dtype=bool)
        for a, b in zip(w0, w1):
            mask = mask | ((self.wlen>a) & (self.wlen<b))
        spec = self._copy()
        spec.flux[~mask] = np.nan
        chip_mask = np.sum(np.isnan(spec.flux), axis=1) < 0.9 * spec.Npix
        # remove fully masked orders
        return self._copy(self.wlen[chip_mask], self.flux[chip_mask], 
                          self.err[chip_mask])


    def remove_order_edge(self, Nedge=20):
        return self._copy(self.wlen[:, Nedge:-Nedge], self.flux[:, Nedge:-Nedge],
                          self.err[:, Nedge:-Nedge], self.Npix-2*Nedge)


    def remove_blackbody(self, teff, debug=False):
        from astropy.modeling import models
        from astropy import units as u
        bb = models.BlackBody(temperature=teff * u.K)
        Nchips = self.wlen.shape[0]
        spec = self._copy()
        norm = []
        for i in range(Nchips):
            flux_nu = bb(self.wlen[i] * u.nm) * np.pi * u.steradian
            norm.append(flux_nu.value)
        norm = np.array(norm)/np.median(norm)
        if debug:
            for i in range(Nchips):
                plt.plot(self.wlen[i], self.flux[i], 'k')
                plt.plot(self.wlen[i], self.flux[i]/norm[i],'r')
            plt.show()
        return self._copy(flux=self.flux/norm, err=self.err/norm)


    def continuum_normalization(self, poly_order=3, debug=False):
        Nchips = self.wlen.shape[0]
        spec = self._copy()
        for i in range(Nchips):
            continuum, _ = su.fit_continuum(self.wlen[i], self.flux[i], poly_order)
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
            spec.flux[i] -= ndimage.gaussian_filter(self.flux[i], sigma=sigma)
        return spec

    def subtract(self, spec):
        if isinstance(spec, self.__class__):
            return self._copy(flux=self.flux-spec.flux)
        # else:
        #     return self._copy(flux=self.flux-np.array(spec))

    def noise_stat(self, spec_model, Nbins=20):
        res = self.subtract(spec_model)
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
