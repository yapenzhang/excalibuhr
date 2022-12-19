
import numpy as np 
import matplotlib.pyplot as plt 


class SPEC2D:

    def __init__(self, wlen, flux, err=None) -> None:
        """
        Object for spectral data
        """
        
        self.wlen = np.array(wlen)
        self.flux = np.array(flux)
        self.err = err
        self.Nchip, self.Nx = self.wlen.shape

        wmin = self.wlen[:,0] 
        indices = np.argsort(wmin)
        self.wlen = self.wlen[indices]
        self.flux = self.flux[indices]
        if self.err is not None:
            self.err = np.array(err)
            self.err = self.err[indices]


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


    def plot_spec1d(self, savename):
        self._set_plot_style()
        nrows = self.Nchip//3
        fig, axes = plt.subplots(nrows=nrows, ncols=1, 
                        figsize=(12,nrows), constrained_layout=True)
        for i in range(nrows):
            ax = axes[i]
            wmin, wmax = self.wlen[i*3][0], self.wlen[i*3+2][-1]
            ymin, ymax = 1, 0
            for j in range(3):
                x, y = self.wlen[i*3+j], self.flux[i*3+j]
                ax.plot(x, y, 'k')
                nans = np.isnan(y)
                vmin, vmax = np.percentile(y[~nans], (10, 90))
                ymin = min(vmin, ymin)
                ymax = max(vmax, ymax)
            ax.set_xlim((wmin, wmax))
            ax.set_ylim((0.4*vmin, 1.3*vmax))
        axes[-1].set_xlabel('Wavelength (nm)')
        axes[-1].set_ylabel('Flux')
        plt.savefig(savename)
        plt.close(fig)

        if self.err is not None:
            fig, axes = plt.subplots(nrows=nrows, ncols=1, 
                        figsize=(12,nrows), constrained_layout=True)
            for i in range(nrows):
                ax = axes[i]
                wmin, wmax = self.wlen[i*3][0], self.wlen[i*3+2][-1]
                ymin, ymax = 1, 0
                for j in range(3):
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