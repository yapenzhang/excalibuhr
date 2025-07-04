# File: src/excalibuhr/pipeline.py

__all__ = ['CriresPipeline']

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import glob
import time
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as Poly
from multiprocessing import Pool
from astropy.io import fits
from astroquery.eso import Eso
import skycalc_ipy
import excalibuhr.utils as su
from excalibuhr.data import SPEC, SERIES, DETECTOR, wfits
import matplotlib.pyplot as plt 
import functools


def print_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n {func.__name__} runtime: {(end_time - start_time):.1f} s \n")
        return result
    return wrapper


class CriresPipeline:

    """
    Parameters
    ----------
    workpath : str
        Path of the main reduction folder. 
    night: str
        The main folder will have subfolders named by dates of observations. 
        In the folder of each night, there will be subfolders called 
        ``raw``, ``cal``, and ``out``, where the raw data (both science and 
        raw calibration), processed calibration files, and data products are 
        stored respectively.
    obs_mode: str
        The observation mode, either 'nod' or 'stare'.
    clean_start: bool
        Set it to True to remove the infomation files when redoing the 
        entire reduction.
    num_processes: int
        number of parallel processes for processing nodding and extraction
    """

    def __init__(self, workpath, night, 
                 obs_mode = 'nod',
                 clean_start = False,
                 num_processes = 4):

        self._print_section(
            f"Run pipeline for Night: {night}", bound_char="=")
        
        self.workpath = os.path.abspath(workpath)
        self.night = night
        self.num_processes = num_processes
        self.nightpath = os.path.join(self.workpath, self.night)
        self.rawpath = os.path.join(self.workpath, self.night, "raw")
        self.calpath = os.path.join(self.workpath, self.night, "cal")
        self.outpath = os.path.join(self.workpath, self.night, "out")
        self.framepath = os.path.join(self.outpath, "frame")
        self.combpath = os.path.join(self.outpath, "combined")
        self.calib_file = os.path.join(self.nightpath, "calib_info.txt") 
        self.header_file = os.path.join(self.nightpath, "header_info.txt")
        self.product_file = os.path.join(self.nightpath, "product_info.txt")
        self.gain = [2.15, 2.19, 2.0]
        self.pix_scale = 0.056 #arcsec
        self.trace_offset = 0

        self.obs_mode = obs_mode.upper()

        print(f"Data reduction folder: {self.nightpath}")

        self.header_keys = {
                    'key_filename': 'ORIGFILE',
                    'key_target_name': 'OBJECT',
                    'key_dtype':'ESO DPR TYPE',
                    'key_catg': 'ESO DPR CATG',
                    'key_mjd': 'MJD-OBS',
                    'key_obsdate': 'DATE-OBS',
                    'key_ra': 'RA',
                    'key_dec': 'DEC',
                    'key_airmass': 'ESO TEL AIRM END',
                    'key_seeing': 'ESO TEL AMBI FWHM END',
                    'key_DIT': 'ESO DET SEQ1 DIT',
                    'key_NDIT': 'ESO DET NDIT',
                    'key_nodpos': 'ESO SEQ NODPOS',
                    'key_nexp_per_nod': 'ESO SEQ NEXPO',
                    'key_nabcycle': 'ESO SEQ NABCYCLES',
                    'key_jitter': 'ESO SEQ JITTERVAL',
                    'key_slitlen': 'ESO INS SLIT1 LEN',
                    'key_slitwid': 'ESO INS SLIT1 NAME',
                    'key_wave_min': 'ESO INS WLEN BEGIN', 
                    'key_wave_max': 'ESO INS WLEN END', 
                    'key_wave_cen': 'ESO INS WLEN CENY', 
                    'key_wlen': 'ESO INS WLEN ID',
                    'key_caltype': 'CAL TYPE',
                    'key_snr': 'SNR',
                            }
        for par in self.header_keys.keys():
            setattr(self, par, self.header_keys[par])

        # self.detlin_path = 'cr2res_cal_detlin_coeffs.fits'

        # in case redo the entire reduction 
        if clean_start:
            if os.path.isfile(self.header_file):
                os.remove(self.header_file)
            if os.path.isfile(self.calib_file):
                os.remove(self.calib_file)
            if os.path.isfile(self.product_file):
                os.remove(self.product_file)
            if os.path.exists(self.calpath):
                shutil.rmtree(self.calpath)
            if os.path.exists(self.outpath):
                shutil.rmtree(self.outpath)

        # Create the directories if they do not exist
        if not os.path.exists(os.path.join(self.workpath, self.night)):
            os.makedirs(os.path.join(self.workpath, self.night))

        if not os.path.exists(self.calpath):
            os.makedirs(self.calpath)

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        if not os.path.exists(self.rawpath):
            os.makedirs(self.rawpath)

        if not os.path.exists(self.framepath):
            os.makedirs(self.framepath)

        if not os.path.exists(self.combpath):
            os.makedirs(self.combpath)

        # If present, read the info files
        if os.path.isfile(self.header_file):
            print("Reading header data from header_info.txt")
            self.header_info = pd.read_csv(self.header_file, sep=';')
        else:

            self.header_info = None 

        if os.path.isfile(self.calib_file):
            print("Reading calibration information from calib_info.txt")
            self.calib_info = pd.read_csv(self.calib_file, sep=';')
        else:
            self.calib_info = None

        if os.path.isfile(self.product_file):
            print("Reading product information from product_info.txt")
            self.product_info = pd.read_csv(self.product_file, sep=';')
        else:
            self.product_info = None
            prod_dict = {}
            keywords = self.header_keys.values()
            for key_item in keywords:
                prod_dict[key_item] = []
            prod_dict[self.key_caltype] = []
            df_prod = pd.DataFrame(data=prod_dict)
            df_prod.to_csv(self.product_file, index=False, sep=';')


    def download_rawdata_eso(self, login, **filters):
        """
        Method for downloading raw data from eso archive using `astroquery`.

        Parameters
        ----------
        login : str
            username to login to the ESO User Portal Services
        filters: 
            optional parameters for data filtering, e.g. `prog_id='...'`, `target='...'`
        """

        self._print_section("Downloading rawdata from ESO arhive")

        print('Night: '+ self.night)
        for key in filters.keys():
            print(key + ': '+ filters[key])

        eso = Eso()
        eso.login(login)
        table = eso.query_instrument(
            'crires', column_filters={'night': self.night, **filters}
            ) 
        data_files = eso.retrieve_data(
            table['DP.ID'], destination=self.rawpath, 
            continuation=False, with_calib='raw', 
            request_all_objects=True, unzip=False)
        os.chdir(self.rawpath)
        for filename in glob.glob("*.xml"):
            os.remove(filename)
        for filename in glob.glob("*.txt"):
            os.remove(filename)
        try:
            os.system("uncompress *.Z")
        except:
            raise OSError("uncompress not found. Please install gzip \
                or ncompress, finish uncompress manually, and proceed \
                to next steps.")
        os.chdir(self.workpath)


    def extract_header(self):
        """
        Method for extracting header information of raw data to a 
        ``pandas.DataFrame`` and a cvs text file.
        """

        self._print_section("Extracting Observation details")

        print("Extracting header details to `header_info.txt`")
        keywords = self.header_keys.values() 
        
        raw_files = sorted(Path(self.rawpath).glob("*.fits"))

        # Dictionary to store the header info
        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:

            header = fits.getheader(file_item)

            # Rename files for better readability
            if self.key_filename in header:
                shutil.move(file_item, 
                    os.path.join(self.rawpath, header[self.key_filename]))

            # Add header value to the dictionary
            for key_item in keywords:
                if key_item in header:
                    header_dict[key_item].append(header[key_item])
                else:
                    header_dict[key_item].append(None)

        # Save the dictionary as a csv-file
        self.header_info = pd.DataFrame(data=header_dict)
        self.header_info.to_csv(self.header_file, index=False, sep=';')


    def _add_to_calib(self, file, cal_type):
        """
        Internal method for adding details of processed calibration files
        to the DataFrame and text file.

        Parameters
        ----------
        file : str
            filename to be added to the table
        cal_type: str 
            type of the calibration file, e.g. `DARK_MASTER`, `FLAT_MASTER`
        """
        print(f"{cal_type}: cal/{file}")
        header = fits.getheader(os.path.join(self.calpath, file))

        calib_dict = {}
        keywords = self.header_keys.values()
        for key_item in keywords:
            calib_dict[key_item] = [header.get(key_item)]
        calib_dict[self.key_caltype] = [cal_type]
        calib_dict[self.key_filename] = [file]
        calib_append = pd.DataFrame(data=calib_dict)
        
        if self.calib_info is None:
            self.calib_info = calib_append
        else:
            self.calib_info = pd.concat([self.calib_info, calib_append], 
                                    ignore_index=True)
        
        self.calib_info.to_csv(self.calib_file, index=False, sep=';')


    def _add_to_product(self, file, prod_type, snr=None):
        """
        Internal method for adding details of data products
        to the DataFrame and text file.

        Parameters
        ----------
        file : str
            filename to be added to the table
        cal_type: str 
            type of the data product, e.g. `NOD_FRAME`, `Extr1D_PRIMARY`

        Returns
        -------
        NoneType
            None
        """

        print(f"Output file -> {prod_type}: out/{file}")
        if file[-4:] == 'fits':
            header = fits.getheader(os.path.join(self.outpath, file))
        else:
            header = {}

        calib_dict = {}
        keywords = self.header_keys.values()
        for key_item in keywords:
            calib_dict[key_item] = [header.get(key_item)]
        calib_dict[self.key_caltype] = [prod_type]
        calib_dict[self.key_filename] = [file]
        calib_dict[self.key_snr] = [snr]
        calib_append = pd.DataFrame(data=calib_dict)
        calib_append.to_csv(self.product_file, index=False, mode='a', header=False, sep=';')
        

    def _print_section(self, sect_title, bound_char = "-", 
                        extra_line = True):
        """
        Internal method for printing a section title.

        Parameters
        ----------
        sect_title : str
            Section title.
        bound_char : str
            Boundary character for around the section title.
        extra_line : bool
            Extra new line at the beginning.
        """

        if extra_line:
            print("\n" + len(sect_title) * bound_char)
        else:
            print(len(sect_title) * bound_char)

        print(sect_title)
        print(len(sect_title) * bound_char + "\n")


    def _download_archive(self, dpr_type, det_dit=None):
        """
        Internal method for downloading data from the ESO Science
        Archive.
        From pycrires (see https://pycrires.readthedocs.io)


        Parameters
        ----------
        dpr_type : str
            The ``DPR.TYPE`` of the data that should be downloaded.
        det_dit : float, None
            The detector integration time (DIT) in case
            ``dpr_type="DARK"``. Can be set to ``None`` otherwise.
        """

        from astropy import units as u
        from astropy import time as t
        while True:
            if det_dit is None:
                download = input(
                    f"Could not find data with DPR.TYPE={dpr_type}. "
                    f"Try to download from the ESO Science Archive "
                    f"Facility (y/n)? "
                )

            else:
                download = input(
                    f"There is not {dpr_type} data with DIT = "
                    f"{det_dit} s. Try to download from the "
                    f"ESO Science Archive Facility (y/n)? "
                )

            if download in ["y", "n", "Y", "N"]:
                break

        if download in ["y", "Y"]:
            indices = self.header_info[self.key_catg] == "SCIENCE"

            # Wavelength setting
            wlen_id = self.header_info[self.key_wlen].iloc[0]

            # Observing date
            date_obs = self.header_info[indices][self.key_obsdate].iloc[0]

            # Sequence of five days before and after the observation
            time_steps = [0.0, -1.0, 1.0, -2.0, 2, -3.0, 3.0, -4.0, 4.0, 
                          -5.0, 5.0, -6.0, 6.0, -7.0, 7.0, -8.0, 8.0,
                          -9.0, 9.0, -10.0, 10.0]
            obs_time = t.Time(date_obs) + np.array(time_steps) * u.day

            # Login at ESO archive
            user_name = input(
                "What is your username to login to the ESO User Portal Services? "
            )
            eso = Eso()
            eso.login(user_name)

            # Query the ESO archive until the data is found
            data_found = False

            for obs_item in obs_time:
                column_filters = {
                    "night": obs_item.value[:10],
                    "dp_type": dpr_type,
                    "ins_wlen_id": wlen_id,
                }

                if det_dit is not None:
                    column_filters["det_dit"] = det_dit

                table = eso.query_instrument("crires", column_filters=column_filters)

                if table is not None:

                    data_files = eso.retrieve_data(
                        table["DP.ID"],
                        destination=self.rawpath,
                        continuation=False,
                        with_calib="none",
                        request_all_objects=False,
                        unzip=False,
                    )

                    os.chdir(self.rawpath)
                    try:
                        os.system("uncompress *.Z")
                    except:
                        raise OSError("uncompress not found. Please install gzip \
                            or ncompress, finish uncompress manually, and proceed \
                            to next steps.")
                    os.chdir(self.workpath)
                    
                    print("\nThe following files have been downloaded:")
                    for data_item in data_files:
                        print(f"   - {data_item}")

                    data_found = True
                    break

            if data_found:
                print(
                    "\nThe requested data has been successfully downloaded. "
                    "Please rerun the pipeline with `clean_start=True` to ensure "
                    "the new calibration data is included."
                )
                sys.exit(0)

            else:
                raise Exception(
                    f"The requested data was not found in the ESO archive. "
                    f"Please download the DPR.TYPE={dpr_type} data manually "
                    "at http://archive.eso.org/wdb/wdb/eso/crires/form. "
                )

        else:
            warnings.warn(
                f"For best results, please download the suggested "
                f"DPR.TYPE={dpr_type} data from the ESO archive at "
                f"http://archive.eso.org/wdb/wdb/eso/crires/form. "
                f"Continuing without the {dpr_type} data now."
            )



    def _set_plot_style(self):
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


    def _plot_det_image(self, savename, title, data, tw = None, slit = None, x_fpet= None):
        """
        Internal method for plotting detector images.

        Parameters
        ----------
        savename : str
            The name of the file to save the plot to.
        title : str
            The title of the plot.
        data : numpy.ndarray
            The 2d image to plot.
        tw : list, optional
            The trace parameters for plotting the order traces.
        slit : list, optional
            The slit parameters for plotting the slit lines.
        x_fpet : list, optional
            The x-positions of the fpet line positions.
        """
        
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

        self._set_plot_style()
        fig, axes = plt.subplots(nrows=1, ncols=Ndet, 
                                figsize=(Ndet*4,4), constrained_layout=True)
        for i in range(Ndet):
            ax, im = axes[i], data[i]
            nans = np.isnan(im)
            vmin, vmax = np.percentile(im[~nans], (1, 99))
            ax.imshow(im, vmin=vmin, vmax=vmax)
            if not tw is None:
                trace = tw[i]
                yy_trace = su.trace_polyval(xx, trace)
                trace_lower, trace_upper = yy_trace
                for o, (yy_upper, yy_lower) in \
                        enumerate(zip(trace_upper, trace_lower)):
                    ax.plot(xx, yy_upper, 'r')
                    ax.plot(xx, yy_lower, 'r')
                    ax.text(xx[len(xx)//10], np.mean(yy_upper+yy_lower)/2., 
                            f'Order {o}', va='center', color='white')
            if not slit is None:
                trace = tw[i]
                slit_meta, x_fpets = slit[i], x_fpet[i]
                im_subs, yy_indices = su.im_order_cut(im, trace)
                yy_poly = su.slit_polyval(x_fpets, slit_meta)
                for (poly, x_peak, yy) in zip(yy_poly, x_fpets, yy_indices):
                    for x, p in zip(x_peak, poly):
                        x_model = Poly.polyval(yy, p)
                        ax.plot(x_model, yy, ':r', zorder=9)

                # for (yy, x_model, x_peak) in zip(yy_indices, xx_shifts, x_fpets):
                #     for x in x_peak:
                #         xs = x - x_model
                #         ax.plot(xs, yy, ':r', zorder=9)
            ax.set_title(f"Detector {i}", size='large', fontweight='bold')

        plt.suptitle(title)
        plt.savefig(savename[:-4]+'png')
        # plt.show()
        plt.close(fig)


    def _plot_spec_by_order(self, savename, flux, wlen=None, transm_spec=None):
        """
        Internal method for plotting spectra by order.

        Parameters
        ----------
        savename : str
            The name of the file to save the plot to.
        flux : list
            The list of spectra for each order and detector.
        wlen : list, optional
            The list of wavelength arrays for each order and detector.
        transm_spec : numpy.ndarray, optional
            The transmission spectrum with wavelength in the first column and flux in the second column.
        """
        flux = np.array(flux)
        Ndet, Norder, Nx = flux.shape
        self._set_plot_style()
        fig, axes = plt.subplots(nrows=Norder, ncols=Ndet, sharey='row',
                        figsize=(6*Ndet,1.5*Norder), constrained_layout=True)
        for i in range(Norder):
            ymin, ymax = 1e8, 0
            for d in range(Ndet):
                ax = axes[Norder-1-i, d]
                y = flux[d, i]
                nans = np.isnan(y)
                vmin, vmax = np.percentile(y[~nans], (1, 99))
                ymin = min(vmin, ymin)
                ymax = max(vmax, ymax)
                if wlen is None:
                    ax.plot(y, 'k')
                    ax.set_xlim((0, Nx))
                else:
                    xx = np.array(wlen)[d, i]
                    ax.plot(xx, y, 'k', label='CRIRES obs.')
                    ax.set_xlim((xx[0], xx[-1]))
                    if transm_spec is not None:
                        indices = (transm_spec[:,0]>xx[0]) & \
                                  (transm_spec[:,0]<xx[-1])
                        ax.plot(transm_spec[:,0][indices], 
                                transm_spec[:,1][indices]*vmax, 
                                color='orange',
                                label='Telluric template')
            ax.set_ylim((0.8*ymin, 1.1*ymax))
        
        for d in range(Ndet):    
            axes[0,d].set_title(f"Detector {d}", size='large', 
                                fontweight='bold')
        
        for i in range(Norder):
            ax = axes[Norder-1-i,-1]
            ax.annotate(f"Order {i}", xy=(1.1,0.5),
                    xycoords='axes fraction',
                    xytext=(0,0), 
                    textcoords='offset points',
                    fontweight='bold',
                    size='large', ha='right', va='center',
                    rotation=90)
        if wlen is None:
            axes[-1,1].set_xlabel('Pixel')
        else:
            axes[-1,1].set_xlabel('Wavelength (nm)')
        axes[-1,0].set_ylabel('Flux')
        if transm_spec is not None:
            axes[-1,-1].legend()
        plt.savefig(savename+'.png')
        # plt.show()
        plt.close(fig)


    def _plot_extr2d_model(self, savename):
        """
        Internal method for plotting 2d intermediate data.

        Parameters
        ----------
        savename : str
            The name of the file to save the plot to.
        """
        self._set_plot_style()
        data = np.load(savename+'.npz')
        D = data['data'][0]
        P = data['data'][2]
        id_det = data['id_det']
        id_order = data['id_order']
        Ndet, Norder = id_det.shape[0]+1, id_order.shape[1]+1
        fig, axes = plt.subplots(nrows=Norder*2, ncols=Ndet, 
                        figsize=(2*Norder, 14), sharex=True, sharey=True,  
                        constrained_layout=True)
        D_det = np.split(D, id_det)
        P_det = np.split(P, id_det)
        for i in range(Ndet):
            D_order = np.split(D_det[i], id_order[i])
            P_order = np.split(P_det[i], id_order[i])
            rows_crop = 0
            for o in range(0, 2*Norder, 2):
                ax_d, ax_m = axes[Norder*2-o-2, i], axes[Norder*2-o-1, i] 
                data, model = D_order[o//2], P_order[o//2]
                rows_crop = max(data.shape[0], rows_crop)
                if data.size != 0:
                    nans = np.isnan(data)
                    vmin, vmax = np.percentile(data[~nans], (5, 95))
                    ax_d.imshow(data, vmin=vmin, vmax=vmax, aspect='auto')
                    ax_m.imshow(model, vmin=0, vmax=np.max(model), aspect='auto')
                ax_d.set_title(f"Order {o//2}")
                # ax_d.set_title(r"Order {0}, $\chi_r^2$: {1:.2f}".format(
                #                         o//2, chi2[i, o//2]))
            axes[-1,i].set_xlabel(f"Detector {i}", size='large', fontweight='bold')
        axes[0,0].set_ylim((0, rows_crop))
        plt.savefig(savename+'.png')
        # plt.show()
        plt.close(fig)


    @print_runtime
    def cal_dark(self, clip=5, combine_mode='median'):
        """
        Method for combining dark frames per DIT while producing readout noise and bad pixel map. 

        Parameters
        ----------
        clip : int
            sigma clipping threshold for rejecting bad pixels
        combine_mode: str
            the way of combining raw dark frames by `mean` or `median`; median combine by defalut.

        See Also
        --------
        :func:`excalibuhr.utils.util_master_dark`
        """

        self._print_section("Create DARK_MASTER")
        
        indices = (self.header_info[self.key_dtype] == "DARK")

        # Check unique DIT
        unique_dit = set()
        for item in self.header_info[indices][self.key_DIT]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            warnings.warn("No DARK frames found in the raw folder")
            self._download_archive("DARK", 1.427049)
        else:
            print(f"DIT values for DARK: {unique_dit}\n")

        # Create a master dark for each unique DIT
        for item in unique_dit:
            
            indices_dit = indices & (self.header_info[self.key_DIT] == item)
            
            # Store each dark-observation in a list
            dt = []
            for file in self.header_info[indices_dit][self.key_filename]:
                with fits.open(os.path.join(self.rawpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))
            
            # Per detector, median-combine the darks
            # determine the bad pixels and readout noise
            master, rons, badpix = su.util_master_dark(dt, badpix_clip=clip,
                                    combine_mode=combine_mode)
            
            print("\n Output files:")
            # Save the master dark, read-out noise, and bad-pixel maps
            file_name = os.path.join(self.calpath, 
                            f'DARK_MASTER_DIT{item}.fits')
            wfits(file_name, ext_list={"FLUX": master}, header=hdr)
            self._add_to_calib(f'DARK_MASTER_DIT{item}.fits', 
                            "DARK_MASTER")
            self._plot_det_image(file_name, f"DARK_MASTER, DIT={item:.1f}",
                                 master)
            
            file_name = os.path.join(self.calpath, 
                            f'DARK_RON_DIT{item}.fits')
            wfits(file_name, ext_list={"FLUX": rons}, header=hdr)
            self._add_to_calib(f'DARK_RON_DIT{item}.fits', 
                            "DARK_RON")

            file_name = os.path.join(self.calpath, 
                            f'DARK_BPM_DIT{item}.fits')
            wfits(file_name, ext_list={"FLUX": badpix.astype(int)}, header=hdr)
            self._add_to_calib(f'DARK_BPM_DIT{item}.fits', 
                            "DARK_BPM")

            print(f"DIT {item:.1f} s -> "
                  f"{np.sum(badpix)/badpix.size*100.:.1f}"
                  r"% of pixels identified as bad")

    
    @print_runtime
    def cal_flat_raw(self, clip=5, combine_mode='median'):
        """
        Method for combining raw flat frames per wavelngth setting and producing bad pixel map.

        Parameters
        ----------
        clip : int
            sigma clipping threshold for rejecting bad pixels
        combine_mode: str
            the way of combining raw flat frames by `mean` or `median`; median combine by defalut.

        See Also
        --------
        :func:`excalibuhr.utils.util_master_flat`
        """

        self._print_section("Create FLAT_MASTER")

        indices = self.header_info[self.key_dtype] == "FLAT"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            warnings.warn("No FLAT frames found in the raw folder")
            self._download_archive("FLAT")
        else:
            print(f"Wavelength settings for FLAT: {unique_wlen}\n")

        # Create a master flat for each unique WLEN setting
        for item_wlen in unique_wlen:

            indices_wlen = indices & \
                        (self.header_info[self.key_wlen] == item_wlen)

            # Use the longest DIT for the master flat
            unique_dit = set()
            for item in self.header_info[indices_wlen][self.key_DIT]:
                unique_dit.add(item)

            dit = max(unique_dit)
            
            indices_dit = indices_wlen & (self.header_info[self.key_DIT] == dit) 

            # Select master dark and bad-pixel mask corresponding to DIT
            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") \
                         & (self.calib_info[self.key_DIT] == dit)
            indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") \
                        & (self.calib_info[self.key_DIT] == dit)
            if np.sum(indices_dark) < 1:
                warnings.warn("No DARK frame found with DIT value corresponding to that of FLAT")
                self._download_archive("DARK", dit)
            else:
                file = self.calib_info[indices_dark][self.key_filename].iloc[0]
                dark = fits.getdata(os.path.join(self.calpath, file))
                file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
                badpix = fits.getdata(os.path.join(self.calpath, file))

            # Store each flat-observation in a list
            dt = []
            for file in self.header_info[indices_dit][self.key_filename]:
                with fits.open(os.path.join(self.rawpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))

            if np.sum(indices_dark) < 1:
                dark = np.zeros_like(dt[0])
            # Per detector, median-combine the flats and determine the bad pixels
            master, badpix = su.util_master_flat(dt, dark, 
                            badpix_clip=clip, combine_mode=combine_mode)
            
            print(f"WLEN setting {item_wlen} -> " 
                  f"{np.sum(badpix)/badpix.size*100.:.1f}"
                  r"% of pixels identified as bad")

            print("\n Output files:")
            # Save the master flat and bad-pixel map
            file_name = os.path.join(self.calpath, 
                            f'FLAT_MASTER_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": master}, header=hdr)
            self._add_to_calib(f'FLAT_MASTER_{item_wlen}.fits', "FLAT_MASTER")

            file_name = os.path.join(self.calpath, 
                            f'FLAT_BPM_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": badpix.astype(int)}, header=hdr)
            self._add_to_calib(f'FLAT_BPM_{item_wlen}.fits', 
                            "FLAT_BPM")
            

    def _loop_over_detector(self, util_func, verbose, *dt_list, **kwargs):
        """
        Method for looping over detectors.

        Parameters
        ----------
        util_func: function
            the util function to run
        verbose: bool
            print the progress over detctors
        *dt_list: 
            input data for the function
        **kwargs: dict
            additional parameters for the function 

        Returns
        -------
        results: list
            output of the util function
        """

        results, results_swap = [], []
        for d in range(len(dt_list[0])):
            if verbose:
                print(f"Processing Detector {d}")
            result = util_func(*[dt[d] for dt in dt_list], **kwargs)
            results.append(result)
        
        # swap axes of the resulting list
        if isinstance(results[0], tuple):
            for n in range(len(results[0])):
                results_swap.append([r[n] for r in results])
            return results_swap
        else:
            return results



    @print_runtime
    def cal_flat_trace(self, debug=False):
        """
        Method for identifying traces of spectral order in the master FLAT frame.

        See Also
        --------
        :func:`excalibuhr.utils.order_trace`
        """

        self._print_section("Trace spectral orders")

        indices = self.calib_info[self.key_caltype] == "FLAT_MASTER"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.calib_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        
        # Identify the trace from the master flat of each WLEN setting
        for item_wlen in unique_wlen:
            indices_flat = indices & \
                        (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & \
                        (self.calib_info[self.key_wlen] == item_wlen)

            file = self.calib_info[indices_flat][self.key_filename].iloc[0]
            flat = fits.getdata(os.path.join(self.calpath, file))
            hdr = fits.getheader(os.path.join(self.calpath, file))
            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))
            
            # Fit polynomials to the trace edges
            trace = self._loop_over_detector(
                            su.order_trace, True, flat, bpm, 
                            slitlen=hdr[self.key_slitlen]/self.pix_scale,
                            offset=self.trace_offset,
                            debug=debug)

            print("\n Output files:")
            # Save the polynomial coefficients
            file_name = os.path.join(self.calpath, f'TW_FLAT_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": trace}, header=hdr)
            self._add_to_calib(f'TW_FLAT_{item_wlen}.fits', "TRACE_TW")
            
            self._plot_det_image(file_name, f"FLAT_MASTER_{item_wlen}", 
                            flat, tw=trace)
            

    @print_runtime
    def cal_slit_curve(self, debug=False):
        """
        Method for tracing slit curvature in the Fabry-Pérot Etalon (FPET) calibration frame.
        Determine initial wavelength solution by mapping FPET lines to 
        an evenly spaced wavelength grid.

        See Also
        --------
        :func:`excalibuhr.utils.slit_curve`
        """

        self._print_section("Trace slit curvature")

        # Select the Fabry-Perot etalon calibrations
        indices_fpet = self.header_info[self.key_dtype] == "WAVE,FPET"
        indices_une = self.header_info[self.key_dtype] == "WAVE,UNE"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices_fpet][self.key_wlen]:
            unique_wlen.add(item)
            
        if len(unique_wlen) == 0:
            warnings.warn("No FPET frames found in the raw folder")
            self._download_archive("WAVE,FPET")

        unique_wlen_une = set()
        for item in self.header_info[indices_une][self.key_wlen]:
            unique_wlen_une.add(item)

        if len(unique_wlen_une) == 0:
            warnings.warn("No FPET frames found in the raw folder")
            self._download_archive("WAVE,UNE")
        
        assert unique_wlen == unique_wlen_une

        # Identify the slit curvature for each WLEN setting
        for item_wlen in unique_wlen:
            indices = indices_une & \
                    (self.header_info[self.key_wlen] == item_wlen)
            file_une = self.header_info[indices][self.key_filename].iloc[0]

            indices = indices_fpet & \
                    (self.header_info[self.key_wlen] == item_wlen)
            file_fpet = self.header_info[indices][self.key_filename].iloc[0]

            dit = self.header_info[indices][self.key_DIT].iloc[0]
            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER")\
                         & (self.calib_info[self.key_DIT] == dit)
            indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") \
                         & (self.calib_info[self.key_DIT] == dit)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & \
                         (self.calib_info[self.key_wlen] == item_wlen)

            if np.sum(indices_dark) < 1:
                warnings.warn(f"No MASTER DARK frame found with the DIT value {dit}s corresponding to that of FPET frame")
                self._download_archive("DARK", dit)
            # if np.sum(indices_tw) < 1:
            #     raise RuntimeError(f"No order trace (TRACE_TW) found with the WLEN setting {item_wlen} corresponding to that of FPET frame") 

            # Read the trace-wave, master dark and bad-pixel mask
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))

            file = self.calib_info[indices_dark][self.key_filename].iloc[0]
            dark = fits.getdata(os.path.join(self.calpath, file))

            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))

            # Dark-subtract the une lamp observation
            with fits.open(os.path.join(self.rawpath, file_une)) as hdu:
                hdr = hdu[0].header
                une = np.array([hdu[i].data for i in range(1, len(hdu))]) - dark

            # correct vertical strips due to readout artifacts
            result = self._loop_over_detector(su.readout_artifact, False,
                                une, une, bpm, tw)
            une, _ = result 

            wlen_mins, wlen_maxs = [], []
            # Dark-subtract the fpet observation
            with fits.open(os.path.join(self.rawpath, file_fpet)) as hdu:
                hdr = hdu[0].header
                fpet = np.array([hdu[i].data for i in range(1, len(hdu))]) - dark
                
                # Store the minimum and maximum wavelengths
                # of each order {j} in each detector {i}.
                for i in range(1, len(hdu)):
                    wlen_min, wlen_max = [], []
                    header = hdu[i].header
                    for j in range(1,11): # maximum 10 orders possible
                        # orders that largely fall out of detectors are ignored
                        if float(header[self.key_wave_cen+str(j)]) > 80 and \
                           float(header[self.key_wave_cen+str(j)]) < 2000:
                            wlen_min.append(header[self.key_wave_min+str(j)])
                            wlen_max.append(header[self.key_wave_max+str(j)])

                    wlen_mins.append(wlen_min)
                    wlen_maxs.append(wlen_max)

            # Assess the slit curvature and wavelengths along the orders
            slit = self._loop_over_detector(su.slit_curve, True,
                            fpet, une, bpm, tw, wlen_mins, wlen_maxs,
                            wlen_id=item_wlen,
                            debug=debug)
            meta, x_fpet, wlen = slit

            print("\n Output files:")
            # Save the polynomial coefficients describing the slit curvature 
            # and an initial wavelength solution
            file_name = os.path.join(self.calpath, 
                            f'SLIT_TILT_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": meta}, header=hdr)
            self._add_to_calib(f'SLIT_TILT_{item_wlen}.fits', "SLIT_TILT")

            self._plot_det_image(file_name, f"FPET_{item_wlen}", 
                            fpet, tw=tw, slit=meta, x_fpet=x_fpet)

            file_name = os.path.join(self.calpath, 
                            f'INIT_WLEN_{item_wlen}.fits')
            wfits(file_name, ext_list={"WAVE": wlen}, header=hdr)
            self._add_to_calib(f'INIT_WLEN_{item_wlen}.fits', "INIT_WLEN")
            

    @print_runtime
    def cal_flat_norm(self, debug=False):
        """
        Method for creating normalized flat field and 
        extracting blaze function in each order.

        See Also
        --------
        :func:`excalibuhr.utils.master_flat_norm`
        """

        self._print_section("Normalize flat; Extract blaze")

        indices = self.calib_info[self.key_caltype] == "FLAT_MASTER"

        indices = self.calib_info[self.key_caltype] == "FLAT_MASTER"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.calib_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        
        # Identify the trace from the master flat of each WLEN setting
        for item_wlen in unique_wlen:
            indices_flat = indices & \
                        (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & \
                        (self.calib_info[self.key_wlen] == item_wlen)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & \
                         (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") \
                          & (self.calib_info[self.key_wlen] == item_wlen)

            # Read in the trace-wave, bad-pixel, slit-curvature, and flat files
            file = self.calib_info[indices_flat][self.key_filename].iloc[0]
            flat = fits.getdata(os.path.join(self.calpath, file))
            hdr = fits.getheader(os.path.join(self.calpath, file))
            
            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))
            
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))
            
            file = self.calib_info[indices_slit][self.key_filename].iloc[0]
            slit = fits.getdata(os.path.join(self.calpath, file))
            
            # Normalize the master flat with the order-specific blaze functions
            result = self._loop_over_detector(su.master_flat_norm, True,
                            flat, bpm, tw, slit, 
                            slitlen=hdr[self.key_slitlen]/self.pix_scale,
                            debug=debug)
            flat_norm, blazes, trace_update = result

            print("\n Output files:")
            file_name = os.path.join(self.calpath, 
                                    f'FLAT_NORM_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": flat_norm}, header=hdr)
            self._add_to_calib(f'FLAT_NORM_{item_wlen}.fits', "FLAT_NORM")
            self._plot_det_image(file_name, f"FLAT_NORM_{item_wlen}", 
                            flat_norm, tw=trace_update)

            file_name = os.path.join(self.calpath, f'BLAZE_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": blazes}, header=hdr)
            self._add_to_calib(f'BLAZE_{item_wlen}.fits', "BLAZE")
            self._plot_spec_by_order(file_name[:-5], blazes) 

            file_name = os.path.join(self.calpath, f'TW_FLAT_{item_wlen}.fits')
            wfits(file_name, ext_list={"FLUX": trace_update}, header=hdr)
        

    @print_runtime
    def obs_nodding(self):
        """
        Method for calibrating and processing science frames. 
        If `obs_mode='nod'`, it subtracts the AB nodding pairs to remove sky background. 
        If `obs_mode='stare'`, each frame will be treated separately and 
        the sky background removal will be performed at a later stage. 
        """

        self._print_section(f"Process {self.obs_mode} frames")

        # Select the science observations
        indices = self.header_info[self.key_catg] == "SCIENCE"

        # Check unique targets
        unique_target = set()
        for item in self.header_info[indices][self.key_target_name]:
            unique_target.add(item)
        if len(unique_target) == 0:
            raise Exception("No SCIENCE data found in the raw folder")
        else:
            print(f"Targets: {unique_target}")

        # Open the read-out noise file
        indices_ron = (self.calib_info[self.key_caltype] == "DARK_RON") 
        file_ron = self.calib_info[indices_ron][self.key_filename].iloc[0]
        ron = fits.getdata(os.path.join(self.calpath, file_ron))

        # initialize a Pool for parallel
        with Pool(processes=self.num_processes) as pool:
            pool_jobs = []
            # Loop over each target
            for object in unique_target:
                print(f"Processing target: {object}")

                indices_obj = indices & \
                        (self.header_info[self.key_target_name] == object)

                # Check unique WLEN setting
                unique_wlen = set()
                for item in self.header_info[indices_obj][self.key_wlen]:
                    unique_wlen.add(item)

                # Loop over each WLEN setting
                for item_wlen in unique_wlen:
                    
                    indices_wlen = indices_obj & \
                                (self.header_info[self.key_wlen] == item_wlen)
                    
                    # Check unique DIT
                    unique_dit = set()
                    for item in self.header_info[indices_wlen][self.key_DIT]:
                        unique_dit.add(item)

                    # Select the corresponding calibration files
                    indices_flat = (self.calib_info[self.key_caltype] == "FLAT_NORM") \
                                & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") \
                            & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") \
                            & (self.calib_info[self.key_wlen] == item_wlen)
                    # indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") \
                    #                & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") \
                                & (self.calib_info[self.key_wlen] == item_wlen)

                    file = self.calib_info[indices_flat][self.key_filename].iloc[0]
                    flat = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
                    bpm = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_tw][self.key_filename].iloc[0]
                    tw = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_slit][self.key_filename].iloc[0]
                    slit = fits.getdata(os.path.join(self.calpath, file))
                
                    # Loop over each DIT
                    for item_dit in unique_dit:
                        print(f"Wavelength setting: {item_wlen}, "
                            f"DIT value: {item_dit} s.")

                        indices_nod_A = indices_wlen & \
                                (self.header_info[self.key_DIT] == item_dit) & \
                                (self.header_info[self.key_nodpos] == 'A')
                        indices_nod_B = indices_wlen & \
                                (self.header_info[self.key_DIT] == item_dit) & \
                                (self.header_info[self.key_nodpos] == 'B')
                        df_nods = self.header_info[indices_nod_A | indices_nod_B]\
                                        .sort_values(self.key_filename)

                        nod_a_count = sum(indices_nod_A)
                        nod_b_count = sum(indices_nod_B)
                        if nod_a_count == nod_b_count:
                            print(f"Number of AB pairs: {nod_a_count}")
                        else:
                            print(f"Number of A and B files: {nod_a_count, nod_b_count}")

                        if self.header_info[indices_nod_A][self.key_nabcycle].iloc[0] == 0 \
                                            or self.obs_mode == 'STARE':
                            # staring mode
                            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") \
                                        & (self.calib_info[self.key_DIT] == item_dit)

                            if np.sum(indices_dark) < 1:
                                warnings.warn(f"No MASTER DARK frame found with the DIT value {item_dit}s corresponding to that of science data")
                                self._download_archive("DARK", item_dit)
                            
                            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") \
                                        & (self.calib_info[self.key_DIT] == item_dit)
                            file = self.calib_info[indices_dark][self.key_filename].iloc[0]
                            dark = fits.getdata(os.path.join(self.calpath, file)) 

                            indices_ron = (self.calib_info[self.key_caltype] == "DARK_RON") \
                                        & (self.calib_info[self.key_DIT] == item_dit)
                            file_ron = self.calib_info[indices_ron][self.key_filename].iloc[0]
                            ron = fits.getdata(os.path.join(self.calpath, file_ron)) 
                            
                            for i in range(df_nods.shape[0]):
                                filename = df_nods[self.key_filename].iloc[i]
                                job = pool.apply_async(self._process_staring, 
                                                    args=(filename, flat, bpm, 
                                                        tw, dark, ron, object, item_wlen))
                                pool_jobs.append(job)
                        else:
                            # nodding mode
                            try:
                                Nexp_per_nod = int(self.header_info[indices_nod_A]\
                                            [self.key_nexp_per_nod].iloc[0])
                            except:
                                # Some headers missing the NEXP key
                                Nexp_per_nod = int(nod_a_count//self.header_info[indices_nod_A][self.key_nabcycle].iloc[0])
                            
                            for i, row in enumerate(range(0, df_nods.shape[0], Nexp_per_nod)):
                                job = pool.apply_async(self._process_nodding, 
                                                    args=(df_nods, i, row, Nexp_per_nod, 
                                                          flat, bpm, tw, ron, object, item_wlen))
                                pool_jobs.append(job)
            
            for job in pool_jobs:
                job.get() 


    def _process_nodding(self, df_nods, i, row, Nexp_per_nod, flat, bpm, 
                             tw, ron, object, item_wlen):
        # check the nodding position of the current frame
        pos = set()
        for p in df_nods[self.key_nodpos].iloc[row]:
        # for p in df_nods[self.key_nodpos].iloc[
        #                 row:row+Nexp_per_nod]:
            pos.add(p)

        if row+(-1)**(i%2)*Nexp_per_nod < df_nods.shape[0]:
            # Select background frames (the opposite nodding position) 
            # correspsonding to the current frame
            pos_bkg = set()
            for p in df_nods[self.key_nodpos].iloc[
                    row+(-1)**(i%2)*Nexp_per_nod: \
                    row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]:
                pos_bkg.add(p)

            bkg_list = [os.path.join(self.rawpath, item) \
                        for item in df_nods[self.key_filename].iloc[
                            row+(-1)**(i%2)*Nexp_per_nod: \
                            row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]
                        ]
        else:
            # in case the nodding cycle was interrupted, use the frame 
            # in the previous cycle as the background image.
            pos_bkg = set()
            for p in df_nods[self.key_nodpos].iloc[
                    row+(-2)*Nexp_per_nod: \
                    row+(-2)*Nexp_per_nod+Nexp_per_nod]:
                pos_bkg.add(p)
                            
            bkg_list = [os.path.join(self.rawpath, item) \
                        for item in df_nods[self.key_filename].iloc[
                            row+(-2)*Nexp_per_nod: \
                            row+(-2)*Nexp_per_nod+Nexp_per_nod]
                        ]

        # make sure not to subtract frames at the same nod position
        if pos == pos_bkg:
            print(df_nods[[self.key_filename, self.key_nodpos]])
            raise Exception("Check if nodding positions of science data are in good order")
        
        dt_list, err_list = [], []
        
        # Loop over the bkg frames and combine them as a bkg image
        for file in bkg_list:
            frame, frame_err = [], []
            with fits.open(file) as hdu:
                ndit = hdu[0].header[self.key_NDIT]
                # Loop over the detectors
                for d in range(1, len(hdu)):
                    frame.append(hdu[d].data)
            # Calculate the detector shot-noise 
            frame_err = su.detector_shotnoise(
                                frame, ron, GAIN=self.gain, NDIT=ndit)
            dt_list.append(frame)
            err_list.append(frame_err)
                            
        # Mean-combine the images if there are multiple 
        # exposures per nod (`Nexp_per_nod`>1).
        dt_bkg, err_bkg = su.combine_frames(dt_list, 
                            err_list, collapse='mean')
                        
        # Select the nod position science image
        # Loop over the observations of the current nod position
        for file in df_nods[self.key_filename].iloc[row:row+Nexp_per_nod]:
            frame, frame_err = [], []
            with fits.open(os.path.join(self.rawpath, file)) as hdu:
                hdr = hdu[0].header
                ndit = hdr[self.key_NDIT]
                # Loop over the detectors
                for d in range(1, len(hdu)):
                    frame.append(hdu[d].data)
                    # Calculate the shot-noise for this detector
                    # For now only consider noise from bkg image
                    # the shot noise from target is not added.
                    frame_err.append(np.zeros_like(hdu[d].data))

            # Subtract the nod-pair from each other
            frame_bkg_cor, err_bkg_cor = su.combine_frames(
                                [frame, -dt_bkg], [frame_err, err_bkg], 
                                collapse='sum')
            # correct vertical strips due to readout artifacts
            result = self._loop_over_detector(su.readout_artifact, False,
                                frame_bkg_cor, err_bkg_cor, bpm, tw)
            frame_bkg_cor, err_bkg_cor = result 
            # Apply the flat-fielding
            frame_bkg_cor, err_bkg_cor = su.flat_fielding(
                                frame_bkg_cor, err_bkg_cor, flat)
            
            file_s = file.split('_')[-1]
            file_name = os.path.join(self.framepath, 
                            f"{self.obs_mode}_"+ object.replace(" ", "") + \
                            f"_{item_wlen}_{file_s}")
            wfits(file_name, ext_list={"FLUX": frame_bkg_cor, 
                                "FLUX_ERR": err_bkg_cor}, header=hdr)

            print(f"\nProcessed file {file_s} at nod position {pos}")
            self._add_to_product('/'.join(file_name.split('/')[-2:]), 
                                f"{self.obs_mode}_FRAME")
            
            self._plot_det_image(file_name, 
                        f"{object}_{self.obs_mode}_FRAME_{item_wlen}", frame_bkg_cor)
    

    def _process_staring(self, file, flat, bpm, tw, dark, ron, object, item_wlen):

        frame, frame_err = [], []
        with fits.open(os.path.join(self.rawpath, file)) as hdu:
            hdr = hdu[0].header
            ndit = hdr[self.key_NDIT]
            slitlen = hdr[self.key_slitlen]
            # Loop over the detectors
            for d in range(1, len(hdu)):
                frame.append(hdu[d].data)
                # For now only consider noise from bkg image
                # the shot noise from target is not added.
                frame_err.append(np.zeros_like(hdu[d].data))

        # Subtract dark frame
        frame_bkg_cor, err_bkg_cor = su.combine_frames(
                            [frame, -dark], [frame_err, ron], 
                            collapse='sum')

        
        # correct vertical strips due to readout artifacts
        result = self._loop_over_detector(su.readout_artifact, False,
                            frame_bkg_cor, err_bkg_cor, bpm, tw)
        frame_bkg_cor, err_bkg_cor = result 

        # Apply the flat-fielding
        frame_bkg_cor, err_bkg_cor = su.flat_fielding(
                            frame_bkg_cor, err_bkg_cor, flat)
        
        file_s = file.split('_')[-1]
        file_name = os.path.join(self.framepath, 
                        f"{self.obs_mode}_"+ object.replace(" ", "") + \
                        f"_{item_wlen}_{file_s}")
        wfits(file_name, ext_list={"FLUX": frame_bkg_cor, 
                            "FLUX_ERR": err_bkg_cor}, header=hdr)

        self._add_to_product('/'.join(file_name.split('/')[-2:]),  
                            f"{self.obs_mode}_FRAME")
        
        self._plot_det_image(file_name, 
                    f"{object}_{self.obs_mode}_FRAME_{item_wlen}", frame_bkg_cor)
    

    @print_runtime
    def obs_nodding_combine(self, combine_mode='mean', clip=3):
        """
        Method for combining multiple nodding exposures at each nodding position A and B.

        Parameters
        ----------
        combine_mode: str
            the way of combining frames: `mean`, `median`, or `weighted`.
            `weighted` applys a weighted average by the mean SNR squared of each 
            exposure, which is estimated using the extratced spectrum of the target. 
        clip: int
            sigma clipping threshold for rejecting bad pixels
        
        See Also
        --------
        :func:`excalibuhr.utils.combine_frames`
        """

        self._print_section("Combine nodding frames")
        
        self.product_info = pd.read_csv(self.product_file, sep=';')
        
        if combine_mode == 'weighted':
            print("Estimate SNR of individual exposures from extracted spectra:")
            savename = 'TMP'
            self.obs_extract(caltype=f"{self.obs_mode}_FRAME", savename=savename)

        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')


        # Select the obs_nodding observations
        indices = (self.product_info[self.key_caltype] == f"{self.obs_mode}_FRAME")

        # Check unique targets
        unique_target = set()
        for item in self.product_info[indices][self.key_target_name]:
            unique_target.add(item)
        if len(unique_target) == 0:
            raise RuntimeError("No reduced nodding frames to combine")

        # Loop over each target
        for object in unique_target:
            print(f"Processing target: {object}")

            indices_obj = indices & \
                    (self.product_info[self.key_target_name] == object)

            # Check unique WLEN setting
            unique_wlen = set()
            for item in self.product_info[indices_obj][self.key_wlen]:
                unique_wlen.add(item)

            # Loop over each WLEN setting
            for item_wlen in unique_wlen:
                
                indices_wlen = indices_obj & \
                            (self.product_info[self.key_wlen] == item_wlen)

                # Loop over the nodding positions
                INT_total = 0.
                for pos in ['A', 'B']:
                    # combine the images for each nodding position
                    
                    indices_pos = indices_wlen & \
                            (self.product_info[self.key_nodpos] == pos)
                    
                    frames, frames_err, snrs = [], [], []
                    # Loop over the observations at each nodding position
                    for j, file in enumerate(
                            self.product_info[indices_pos][self.key_filename]):
                        with fits.open(os.path.join(self.outpath, file)) as hdu:
                            hdr = hdu[0].header
                            # in case of jittering
                            if np.isclose(hdr[self.key_jitter], 0):
                                dt, dt_err = hdu["FLUX"].data, hdu["FLUX_ERR"].data
                            else:
                                # apply integer shift to align frames
                                dt, dt_err = su.align_jitter(
                                    hdu["FLUX"].data, hdu["FLUX_ERR"].data, 
                                    int(np.round(hdr[self.key_jitter]/self.pix_scale)))
                            frames.append(dt)
                            frames_err.append(dt_err)
                        
                        # get signal to noise ratio of the observation
                        if combine_mode == 'weighted':
                            name_tmp = file.split('/')
                            name_tmp[1] = '_'.join(['Extr1D_PRIMARY', savename, name_tmp[1]])
                            indice_snr = (self.product_info[self.key_filename] == '/'.join(name_tmp) )
                            snr = self.product_info[indice_snr][self.key_snr].iloc[0]
                            snrs.append(snr)
                            print(f"{file}: SNR={snr}")
                    
                    print("\nCombining {0:d} frames at nodding".format(j+1),
                         f"position {pos}")

                    if combine_mode == 'weighted':
                        weights = np.square(snrs)
                    else:
                        weights = None

                    combined, combined_err = su.combine_frames(
                                        frames, frames_err, 
                                        collapse=combine_mode, clip=clip,
                                        weights=weights)
                    
                    # Save the combined obs_nodding observation
                    file_name = os.path.join(self.combpath, 
                            "COMBINED_"+ object.replace(" ", "") + \
                            f"_{item_wlen}_{self.obs_mode}_{pos}.fits")
                    wfits(file_name, ext_list={"FLUX": combined, 
                                "FLUX_ERR": combined_err}, header=hdr)
                    self._add_to_product('/'.join(file_name.split('/')[-2:]), 
                            f"{self.obs_mode}_COMBINED")

                    self._plot_det_image(file_name, 
                        f"{object}_{self.obs_mode}_{pos}_COMBINED_{item_wlen}", combined)

                    INT_total += hdr[self.key_DIT]*hdr[self.key_NDIT]*(j+1)/3600.
                
                print(f"On-target time for {object} with wavelength setting {item_wlen}: {INT_total:.2f} hrs \n") 


    @print_runtime
    def obs_extract(self, caltype='NOD_COMBINED', 
                          object=None,
                          savename='',
                          peak_frac=None, companion_sep=None, 
                          remove_star_bkg=False,
                          remove_sky_bkg=False,
                          aper_prim=20, aper_comp=10, 
                          extract_2d=True,
                          extr_level=0.9,
                          debug=False):    
        """
        Method for extracting 1D spectrum from the 2D science frames. It works on either 
        combined frames or individual exposures, as specified by the `caltype` option.
        It extracts all targets taken during the night if not specified with the `object` parameter.  

        Parameters
        ----------
        caltype: str
            Label of the file type to worked on: `NODDING_COMBINED` or `NODDING_FRAME`.
        object: str
            name of the target to extract 1D spectrum. If not specified, all targets will be extracted.
        savename: str
            customed label for the extratced spectrum (optional).
        peak_frac: dict
            the fraction of target signal along the slit at A and B nodding position 
            (a fraction of 0 is at the bottom of the slit and 1 at the top).
            If not specified, the code will determine the peak location from the whitelight PSF of the data. 
        companion_sep: float
            separation of the companion in arcsec
        remove_star_bkg: bool
            whether to remove the stellar background near the resolved companion using polynomials 
            at each wavelength channel.
        remove_sky_bkg: bool
            whether to remove the sky background. This will be performed for the observations in 'STARE' mode.
        aper_prim: int
            half of extraction aperture in pixel for the primary star
        aper_comp: int
            half of extraction aperture in pixel for the companion
        extr_level: float
            the extraction aperture will be automatically refined to include a sepcified fraction of 
            the total flux in the derived PSF of the data. By default it encompasses 90% of the flux.
        extract_2d: bool
            if True, the curvature of the trace in the 2D images will be corrected and 
            the intermediate 2D data will be saved to .npz files. 
        debug : bool
            generate plots for debugging.

        See Also
        --------
        :func:`excalibuhr.utils.extract_spec`
        :func:`excalibuhr.utils.optimal_extraction`
        """

        self._print_section("Extract spectra")

        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')


        # Select the type of observations we want to work with
        indices = (self.product_info[self.key_caltype] == caltype)

        unique_target = set()
        if object is not None:
            unique_target.add(object)
        else:
            # Check all unique targets
            for item in self.product_info[indices][self.key_target_name]:
                unique_target.add(item)
            if len(unique_target) == 0:
                raise RuntimeError("No reduced frames to extract")

        # initialize a Pool for parallel
        with Pool(processes=self.num_processes) as pool:
            pool_jobs = []
            # Loop over each target
            for object in unique_target:
                print(f"Processing target: {object} \n")

                indices_obj = indices & \
                        (self.product_info[self.key_target_name] == object)

                # Check unique WLEN setting
                unique_wlen = set()
                for item in self.product_info[indices_obj][self.key_wlen]:
                    unique_wlen.add(item)

                # Loop over each WLEN setting
                for item_wlen in unique_wlen:
                    # print(f"Wavelength setting: {item_wlen}")
                    
                    indices_wlen = indices_obj & \
                                (self.product_info[self.key_wlen] == item_wlen)
                
                    # Select the corresponding calibration files
                    indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") \
                                & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") \
                            & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") \
                                & (self.calib_info[self.key_wlen] == item_wlen)
                    indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") \
                                & (self.calib_info[self.key_wlen] == item_wlen)
                    # indices_wave = (self.calib_info[self.key_caltype] == "INIT_WLEN") & \
                    #                (self.calib_info[self.key_wlen] == item_wlen)

                    file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
                    bpm = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_tw][self.key_filename].iloc[0]
                    tw = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_slit][self.key_filename].iloc[0]
                    slit = fits.getdata(os.path.join(self.calpath, file))
                    file = self.calib_info[indices_blaze][self.key_filename].iloc[0]
                    blaze = fits.getdata(os.path.join(self.calpath, file))

                    
                    # Loop over each observation
                    # if object == std_object:
                    #     for file in self.product_info[indices_wlen][self.key_filename]:
                    #         job = pool.apply_async(self._process_extraction, 
                    #                             args=(file, bpm, tw, slit, blaze, 
                    #                                 peak_frac, aper_prim, aper_comp, 
                    #                                 None, extract_2d, 
                    #                                 remove_bkg, extr_level,
                    #                                 savename, debug))
                    #         pool_jobs.append(job)
                    # else:
                    for file in self.product_info[indices_wlen][self.key_filename]:
                        job = pool.apply_async(self._process_extraction, 
                                            args=(file, caltype.split('_')[1], 
                                                  bpm, tw, slit, blaze, 
                                                peak_frac, aper_prim, aper_comp, 
                                                companion_sep, extract_2d, extr_level,
                                                remove_star_bkg, remove_sky_bkg,
                                                savename, debug))
                        pool_jobs.append(job)
            
            for job in pool_jobs:
                job.get() 

    def _process_extraction(self, file, filetype, bpm, tw, slit, blaze, 
                            peak_frac, aper_prim, aper_comp, 
                            companion_sep, extract_2d, extr_level,
                            remove_star_bkg, remove_sky_bkg, 
                            savename, debug):
        
        with fits.open(os.path.join(self.outpath, file)) as hdu:
            hdr = hdu[0].header
            dt = hdu["FLUX"].data
            dt_err = hdu["FLUX_ERR"].data
        pos = hdr[self.key_nodpos]
        slitlen = hdr[self.key_slitlen]
        ndit = hdr[self.key_NDIT]
           
        if peak_frac is not None:
            f0 = int(peak_frac[pos]*slitlen/self.pix_scale)
        else:
            f0 = None

        # Extract 1D (and 2D) spectrum of the target
        result = self._loop_over_detector(
                        su.extract_spec, False,
                        dt, dt_err, bpm, tw, slit, blaze, 
                        self.gain, NDIT=ndit, extract_2d=extract_2d,
                        cen0=f0, remove_star_bkg=remove_star_bkg,
                        remove_sky_bkg=remove_sky_bkg,
                        extr_level=extr_level,
                        aper_half=aper_prim, debug=debug)
        flux_pri, err_pri, D, V, P = result

        #estimate snr of the primary spectra at the middle order
        mid_order = len(flux_pri[0])//2
        # snr_mid = np.nanmean(np.array(flux_pri)[:, mid_order ,:])
        snr_mid = np.nanmean(np.array(flux_pri)[:, mid_order ,:]/np.array(err_pri)[:, mid_order ,:]).astype(int)

        paths = file.split('/')
        paths[-1] = '_'.join(['Extr1D_PRIMARY', savename, paths[-1]])
        filename = os.path.join(self.outpath, '/'.join(paths))
        wfits(filename, ext_list={"FLUX": flux_pri, "FLUX_ERR": err_pri}, header=hdr)
        if savename == '':
            self._add_to_product('/'.join(paths), f'Extr1D_{filetype}_PRIMARY', snr_mid)
        else:
            self._add_to_product('/'.join(paths), '_'.join(['Extr1D', filetype, savename]), snr_mid)
        self._plot_spec_by_order(filename[:-5], flux_pri)
        
        if extract_2d:
            paths = file.split('/')
            paths[-1] =  '_'.join(['Extr2D_PRIMARY', savename, paths[-1][:-5]])
            filename2d = os.path.join(self.outpath, '/'.join(paths))

            extr2d = DETECTOR(data=[D, V, P], fields=['flux', 'var', 'psf'])
            extr2d.save_extr2d(filename2d)
            extr2d.plot_extr2d_model(filename2d)

            if savename == '':
                self._add_to_product('/'.join(paths)+'.npz', f'Extr2D_{filetype}_PRIMARY')
            else:
                self._add_to_product('/'.join(paths)+'.npz', '_'.join(['Extr2D', filetype, savename]))

        if not companion_sep is None:

            # Extract a 1D spectrum for the secondary
            result = self._loop_over_detector(
                            su.extract_spec, False,
                            dt, dt_err, bpm, tw, slit, blaze, 
                            self.gain, NDIT=ndit,
                            cen0=f0, extract_2d=extract_2d, 
                            companion_sep=companion_sep/self.pix_scale,
                            aper_half=aper_comp, 
                            remove_star_bkg=remove_star_bkg,
                            remove_sky_bkg=remove_sky_bkg,
                            extr_level=extr_level,
                            debug=debug)
            flux_sec, err_sec, D, V, P = result

            paths = file.split('/')
            paths[-1] = '_'.join(['Extr1D_SECONDARY', savename, paths[-1]])
            filename = os.path.join(self.outpath, '/'.join(paths))
            wfits(filename, ext_list={"FLUX": flux_sec, 
                                    "FLUX_ERR": err_sec}, header=hdr)
            if savename == '':
                self._add_to_product('/'.join(paths), f'Extr1D_{filetype}_SECONDARY')
            else:
                self._add_to_product('/'.join(paths), '_'.join(['Extr1D', filetype, savename]))
            self._plot_spec_by_order(filename[:-5], flux_sec)
            
            paths = file.split('/')
            paths[-1] = '_'.join(['Extr2D_SECONDARY', savename, paths[-1][:-5]])
            filename2d = os.path.join(self.outpath, '/'.join(paths))
            
            extr2d = DETECTOR(data=[D, V, P], fields=['flux', 'var', 'psf'])
            extr2d.save_extr2d(filename2d)
            extr2d.plot_extr2d_model(filename2d)

            if savename == '':
                self._add_to_product('/'.join(paths)+'.npz', f'Extr2D_{filetype}_SECONDARY')
            else:
                self._add_to_product('/'.join(paths)+'.npz', '_'.join(['Extr2D', filetype, savename]))
    


    @print_runtime
    def refine_wlen_solution(self, debug=False):
        """
        Method for refining wavelength solution by manimizing 
        chi2 between the spectrum and the telluric transmission
        template on a order-by-order basis.
        The calibrated spectrum of each target is saved to a fits file.

        See Also
        --------
        :func:`excalibuhr.utils.wlen_solution`
        """

        self._print_section("Refine wavelength solution")

        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')

        indices = (self.product_info[self.key_caltype] == 'Extr1D_COMBINED_PRIMARY') 

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.product_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        # get telluric transmission model
        file = os.path.join(self.calpath, "TRANSM_SPEC.fits")
        if not os.path.isfile(file):
            airmass = self.product_info[indices][self.key_airmass].max()
            self.run_skycalc(airmass=airmass)
        tellu = fits.getdata(file)

        for item_wlen in unique_wlen:
            print(f"Calibrating WLEN setting {item_wlen}:")

            indices_wave = (self.calib_info[self.key_caltype] == "INIT_WLEN") \
                         & (self.calib_info[self.key_wlen] == item_wlen)
            file = self.calib_info[indices_wave][self.key_filename].iloc[0]
            wlen_init = fits.getdata(os.path.join(self.calpath, file))
            hdr = fits.getheader(os.path.join(self.calpath, file))
            
            indices_wlen = indices & \
                          (self.product_info[self.key_wlen] == item_wlen)
            
            unique_target = set()
            for item in self.product_info[indices_wlen][self.key_target_name]:
                unique_target.add(item)

            for target in unique_target:

                # select available data of the target
                indices_obj = indices_wlen & \
                            (self.product_info[self.key_target_name] == target)
                
                dt, dt_err = [], []
                for file in self.product_info[indices_obj][self.key_filename]:
                    with fits.open(os.path.join(self.outpath, file)) as hdu:
                        dt.append(hdu["FLUX"].data)
                        dt_err.append(hdu["FLUX_ERR"].data)
                
                # sum available spectra 
                dt, dt_err = su.combine_frames(dt, dt_err, collapse='sum')

                # Convolve telluric model to the instrument resolution
                slit_width = self.product_info[indices_obj][self.key_slitwid].iloc[0]
                if slit_width == "w_0.2":
                    spec_res = 100000.0
                elif slit_width == "w_0.4":
                    spec_res = 50000.0

                tellu_conv = su.SpecConvolve(tellu[:,0], tellu[:,1], 
                                out_res=spec_res, in_res=2e5)
                tellu_conv = np.column_stack((tellu[:,0], tellu_conv))

                wlen_cal = self._loop_over_detector(su.wlen_solution, True,
                            dt, dt_err, wlen_init, transm_spec=tellu_conv, 
                            debug=debug)

                file_name = os.path.join(self.calpath, f'WLEN_{item_wlen}_' \
                                        + '_'.join(target.split())+'.fits')
                hdr[self.key_target_name] = target
                wfits(file_name, ext_list={"WAVE": wlen_cal}, header=hdr)
                self._add_to_calib(f'WLEN_{item_wlen}_' \
                                    + '_'.join(target.split())+'.fits', "CAL_WLEN")

                self._plot_spec_by_order(file_name[:-5], dt, wlen_cal, 
                                        transm_spec=tellu_conv)
        

        self._print_section("Save extracted spectra")

        # get labels containing 'Extr1D'
        all_labels = self.product_info[self.key_caltype].unique()
        data_frame = all_labels[pd.Series(all_labels).str.contains('Extr1D_FRAME', case=False)]
        data_comb = all_labels[pd.Series(all_labels).str.contains('Extr1D_COMBINED', case=False)]

        # mask the detector edges 
        Ncut = 10

        for data_type in [data_frame, data_comb]:
            for label in data_type:
                indices = (self.product_info[self.key_caltype] == label)
                
                # Check unique targets
                unique_target = set()
                for item in self.product_info[indices][self.key_target_name]:
                    unique_target.add(item)

                if len(unique_target) == 0:
                    continue
                
                # Loop over each target
                for target in unique_target:

                    # output filename
                    l = label.split('_')[-1]
                    if data_type == data_comb:
                        file_name = os.path.join(self.combpath, 
                            '_'.join(['SPEC', target.replace(" ", ""), l]))
                    else:
                        file_name = os.path.join(self.framepath, 
                            '_'.join(['SPEC_SERIES', target.replace(" ", ""), l]))
                        
                    indices_obj = indices & \
                        (self.product_info[self.key_target_name] == target)

                    # Check unique WLEN setting
                    unique_wlen = set()
                    for item in self.product_info[indices_obj][self.key_wlen]:
                        unique_wlen.add(item)
                    

                    wlens, specs, errs = [],[],[]
                    for item_wlen in unique_wlen:
                        
                        indices_wlen = indices_obj & \
                                (self.product_info[self.key_wlen] == item_wlen)

                        indices_wave = \
                            (self.calib_info[self.key_caltype] == "CAL_WLEN")\
                            & (self.calib_info[self.key_wlen] == item_wlen) \
                            & (self.calib_info[self.key_target_name] == target)
                        
                        if not np.any(indices_wave):
                            print("No matching wavelength solution to the target.\n")
                            print("The wavelength solution derived from other target is used.\n")
                            indices_wave = \
                                    (self.calib_info[self.key_caltype] == "CAL_WLEN")\
                                    & (self.calib_info[self.key_wlen] == item_wlen)
                            
                            if not np.any(indices_wave):
                                print("No calibrated wavelength solution available.\n")
                                print("Initial wavelength solution is used.\n")
                                indices_wave = \
                                        (self.calib_info[self.key_caltype] == "INIT_WLEN") \
                                        & (self.calib_info[self.key_wlen] == item_wlen)
                            
                        file = self.calib_info[indices_wave][self.key_filename].iloc[0]
                        wlen = fits.getdata(os.path.join(self.calpath, file))
                        wlens.append(wlen)

                        dt, dt_err = [], []
                        for file in self.product_info[indices_wlen][self.key_filename]:
                            with fits.open(os.path.join(self.outpath, file)) as hdu:
                                hdr = hdu[0].header
                                dt.append(hdu["FLUX"].data)
                                dt_err.append(hdu["FLUX_ERR"].data)
                        nframe = len(dt)
                        if data_type == data_comb:
                            # mean-combine each individual frames
                            dt, dt_err = su.combine_frames(dt, dt_err, collapse='mean')
                            self._plot_spec_by_order(file_name+'_'+item_wlen, dt, wlen)

                        specs.append(dt)
                        errs.append(dt_err)
                    
                    wlens = np.array(wlens)
                    npixel = wlens.shape[-1]
                    wlens = np.reshape(wlens, (-1, npixel))
                    wmin = wlens[:,0] 
                    indice_sort = np.argsort(wmin)
                    wlens = wlens[indice_sort]

                    
                    if data_type == data_comb:
                        # reshape spectra in 2D shape: (N_chips, N_pixel)
                        spec_series = np.reshape(specs, (-1, npixel))[indice_sort]
                        err_series = np.reshape(errs, (-1, npixel))[indice_sort]
                        spec_series[:, :Ncut] = np.nan
                        spec_series[:, -Ncut:] = np.nan
                        snr_mid = np.nanmean((spec_series/err_series)[wlens.shape[0]//2])

                    else:
                        # reshape spectra in 3D shape: (N_frames, N_chips, N_pixel)
                        spec_series, err_series = [], []
                        specs, errs = np.array(specs), np.array(errs)
                        for i in range(nframe):
                            spec_series.append(np.reshape(specs[:,i,:,:,:], (-1, npixel))[indice_sort])
                            err_series.append(np.reshape(errs[:,i,:,:,:], (-1, npixel))[indice_sort])
                        spec_series = np.array(spec_series)
                        err_series = np.array(err_series)
                        spec_series[:,:, :Ncut] = np.nan
                        spec_series[:,:, -Ncut:] = np.nan
                        snr_mid = np.nanmean((spec_series/err_series)[:,wlens.shape[0]//2,:], axis=1)


                    wfits(file_name+'.fits', ext_list={"FLUX": spec_series, 
                                                "FLUX_ERR": err_series,
                                                "WAVE": wlens}, 
                                        header=hdr)
                    self._add_to_product('/'.join(file_name.split('/')[-2:])+'.fits', 
                                    '_'.join(['SPEC']+label.split('_')[-2:]))
                    if data_type == data_comb:
                        np.savetxt(file_name+'.dat', np.c_[wlens.flatten(), 
                                                            spec_series.flatten(), 
                                                            err_series.flatten()], 
                                    header="wave flux err")
                    
                    if isinstance(snr_mid, float):
                        print(f"Saved target {target} {l} with wavelength coverage {unique_wlen}; ",
                                f"average S/N ~ {snr_mid:.0f}. \n")
                    else:
                        print(f"Saved target {target} {l} with wavelength coverage {unique_wlen}; average S/N ~ ",
                                np.round(snr_mid).astype(int), " \n")




    def run_skycalc(self, airmass=1.0, pwv=2.5):
        """
        Method for obtaining the telluric transmission template using the Python wrapper of `SkyCalc`:
        https://skycalc-ipy.readthedocs.io 

        Parameters
        ----------
        airmass : float
            airmass 
        pwv : float
            Precipitable water vapor (default: 2.5 mm)
    
        """

        self._print_section("Obtain telluric transmission with SkyCalc")

        # Indices with SCIENCE frames
        indices = self.header_info[self.key_catg] == "SCIENCE"

        # Setup SkyCalc object
        sky_calc = skycalc_ipy.SkyCalc()

        wlen_id = self.header_info[indices][self.key_wlen].iloc[0]
        mjd_start = self.header_info[indices][self.key_mjd].iloc[0]
        ra_mean = self.header_info[self.key_ra][indices].iloc[0]
        dec_mean = self.header_info[self.key_dec][indices].iloc[0]

        sky_calc.get_almanac_data(
            ra=ra_mean,
            dec=dec_mean,
            date=None,
            mjd=mjd_start,
            observatory="paranal",
            update_values=True,
        )

        sky_calc["msolflux"] = 130

        if wlen_id[0] == "Y":
            sky_calc["wmin"] = 500.0  # (nm)
            sky_calc["wmax"] = 1500.0  # (nm)

        elif wlen_id[0] == "J":
            sky_calc["wmin"] = 800.0  # (nm)
            sky_calc["wmax"] = 2000.0  # (nm)

        elif wlen_id[0] == "H":
            sky_calc["wmin"] = 1000.0  # (nm)
            sky_calc["wmax"] = 2500.0  # (nm)

        elif wlen_id[0] == "K":
            sky_calc["wmin"] = 1850.0  # (nm)
            sky_calc["wmax"] = 2560.0  # (nm)

        elif wlen_id[0] == "L":
            sky_calc["wmin"] = 2500.0  # (nm)
            sky_calc["wmax"] = 4500.0  # (nm)

        else:
            raise NotImplementedError(
                f"The wavelength range for {wlen_id} is not yet implemented."
            )

        sky_calc["wgrid_mode"] = "fixed_spectral_resolution"
        sky_calc["wres"] = 5e5
        sky_calc["pwv"] = pwv
        sky_calc['airmass'] = airmass 

        # print(f"  - Wavelength range (nm) = {sky_calc['wmin']} - {sky_calc['wmax']}")
        # print(f"  - lambda / Dlambda = {sky_calc['wres']}")
        # print(f"  - Airmass = {sky_calc['airmass']:.2f}")
        # print(f"  - PWV (mm) = {sky_calc['pwv']}\n")

        # Get telluric spectra from SkyCalc

        print("Get telluric spectrum with SkyCalc...", end="", flush=True)

        wave, trans, _ = sky_calc.get_sky_spectrum(return_type="arrays")

        print(" [DONE]\n")

        transm_spec = np.column_stack((wave.value, trans))
        out_file= os.path.join(self.calpath, "TRANSM_SPEC.fits")
        wfits(out_file, ext_list={"FLUX": transm_spec})



    @print_runtime
    def run_molecfit(self, object=None, data_type=None, wave_range=None, verbose=True):
        """
        Method for running ESO's tool for telluric correction `Molecfit`. 

        Parameters
        ----------
        object: str
            Name of the star whose spectra is used for the telluric fitting. 
            Default is to carry out fitting for every target.
        data_type: str
            Label of the file type to fit for telluric absorption.
            The default value is `SPEC_COMBINED_PRIMARY`, i.e. the primary spectrum.
            It can also be individual frames in time-series observations.
        wave_range: list of tuple
            list of lower and upper limit for wavelength ranges (in um)
            for molecfit fitting
        verbose : bool
            Print output produced by ``esorex``.
        
        See Also
        --------
        :func:`excalibuhr.utils.molecfit`
        """

        self._print_section("Run Molecfit")

        # Create the molecfit directory if it does not exist yet
        self.molpath = os.path.join(self.outpath, "molecfit")
        if not os.path.exists(self.molpath):
            os.makedirs(self.molpath)


        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')

        if data_type is None:
            data_type = 'SPEC_COMBINED_PRIMARY'
        indices = (self.product_info[self.key_caltype] == data_type)

        if not object is None:
            indices = indices & \
                (self.product_info[indices][self.key_target_name] == object)

        for science_file in self.product_info[indices][self.key_filename]:

            name = science_file.split('/')[-1] 
            target = name.split('_')[0]

            if data_type == 'SPEC_COMBINED_PRIMARY':
                dt = SPEC(filename=os.path.join(self.outpath, science_file))
                dt.wlen *= 1e-3
                su.molecfit(self.molpath, dt, wave_range, savename=name, verbose=verbose)

                
                file_best_par = fits.getdata(os.path.join(self.molpath, 'BEST_FIT_PARAMETERS.fits'))
                best_params = fits.getdata(file_best_par, 1)
                np.savetxt(os.path.join(self.molpath, f'BEST_FIT_PARAMETERS_{target}.txt'), 
                                best_params.tolist()[:-2], fmt='%s')

                # save telluric model
                tellu = fits.getdata(os.path.join(self.molpath, 'TELLURIC_DATA.fits'))
                file_name = os.path.join(self.molpath, f'TELLURIC_{target}.fits')
                wfits(file_name, ext_list={"WAVE":tellu['lambda']*1e3, 
                                           "FLUX":tellu['mtrans']}, 
                                header=dt.header)

            elif data_type == 'SPEC_FRAME_PRIMARY':
                dt_series = SERIES(filename=os.path.join(self.outpath, science_file))
                trans_model = []
                for dt in dt_series:
                    dt.wlen *= 1e-3
                    su.molecfit(self.molpath, dt, wave_range, savename=name, verbose=verbose)

                    tellu = fits.getdata(os.path.join(self.molpath, 'TELLURIC_DATA.fits'))
                    trans_model.append(tellu['mtrans'])

                file_name = os.path.join(self.molpath, f'TELLURIC_{target}.fits')
                wfits(file_name, ext_list={"WAVE": tellu['lambda']*1e3, 
                                           "FLUX": trans_model}, 
                                header=dt_series.header)
            
            self._add_to_product(f'molecfit/TELLURIC_{target}.fits', "TELLU_MOLECFIT")



    @print_runtime
    def spec_response_cal(self, object, temp, vsini, vsys, 
                          mask_wave=[(2164, 2169)], debug=False
                        ):
        """
        Method for calibrating spectral shape using standard star observations.

        Parameters
        ----------
        object: str
            Name of the standard star whose spectrum is used for
            the instrument response calibration. 
        temp: int
            effective temperature of the standard star
        vsini: float
            rotational broadenning of the standard star
        vsys: float
            systemic and barycentric  velocity correction for the standard star   
        mask_wave: list of tuple
            list of lower and upper limit for wavelength ranges (in nm) that need to 
            be masked when estimating the isntrument response.
            They are usually Hydrogen lines of the stellar spectrum.

        
        See Also
        --------
        :func:`excalibuhr.utils.instrument_response`
        """

        self._print_section("Spectral response")
        
        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')

        data_type = 'SPEC_COMBINED_PRIMARY'
        indices_std =  (self.product_info[self.key_caltype] == data_type) & \
                        (self.product_info[self.key_target_name] == object)

        std_file = self.product_info[indices_std][self.key_filename].iloc[0]
        std = SPEC(filename=os.path.join(self.outpath, std_file))

        tellu_file = os.path.join(self.calpath, "TRANSM_SPEC.fits")
        if not os.path.isfile(tellu_file):
            airmass = self.product_info[indices_std][self.key_airmass].max()
            self.run_skycalc(airmass=airmass)

        # Convolve telluric model to the instrument resolution
        slit_width = self.product_info[indices_std][self.key_slitwid].iloc[0]
        if slit_width == "w_0.2":
            spec_res = 100000.0
        elif slit_width == "w_0.4":
            spec_res = 50000.0

        tellu = fits.getdata(tellu_file)
        tellu[:,1] = su.SpecConvolve(tellu[:,0], tellu[:,1], 
                        out_res=spec_res, in_res=2e5)

        # # skycalc model is not sufficient, use molecfit model instead
        # indices_tellu =  (self.product_info[self.key_caltype] == 'TELLU_MOLECFIT') & \
        #                 (self.product_info[self.key_target_name] == object)
        # if np.sum(indices_tellu) > 0:
        #     tellu_file = self.product_info[indices_tellu][self.key_filename].iloc[0]
        # with fits.open(os.path.join(self.outpath, tellu_file)) as hdu:
        #     tellu = hdu['FLUX'].data

        response, tellu_std = su.instrument_response(std, tellu, 
                                        temp, vsini, vsys, 
                                        mask_wave, debug=debug)

        file_name = os.path.join(self.calpath, "RESPONSE.dat")
        np.savetxt(file_name, np.c_[np.ravel(std.wlen), response, tellu_std])



    @print_runtime
    def apply_correction(self, use_molecift=False):
        """
        Method for apply telluric and instrument response correction
        using the output from `run_molecfit` and `spec_response_cal`. 
        The corrected spectrum is save to a text file.

        Parameters
        ----------
        use_molecfit: bool
            If True, it uses the telluric model generated by `molecfit`. Otherwise, 
            the standard stellar spectrum is used to correct for the telluric.

        """

        self._print_section("Apply corrections")

        # get updated product info
        self.product_info = pd.read_csv(self.product_file, sep=';')

        # get instrument response
        file_name = os.path.join(self.calpath, "RESPONSE.dat")
        w_std, resp, tellu_std = np.genfromtxt(file_name, unpack=1)

        # get labels containing 'SPEC'
        all_labels = self.product_info[self.key_caltype].unique()
        data_type = all_labels[pd.Series(all_labels).str.contains('SPEC_COMBINED', case=False)]

        for label in data_type:

            indices = (self.product_info[self.key_caltype] == label)
            
            # Check unique targets
            unique_target = set()
            for item in self.product_info[indices][self.key_target_name]:
                unique_target.add(item)

            if len(unique_target) == 0:
                continue
            
            # Loop over each target
            for target in unique_target:

                if use_molecift:
                    # get molecfit telluric model
                    indices_tellu = (self.product_info[self.key_caltype] == 'TELLU_MOLECFIT') & \
                                    (self.product_info[self.key_target_name] == target)
                    if sum(indices_tellu) < 1:
                        raise Exception("No telluric model found")

                    with fits.open(os.path.join(self.outpath, 
                            self.product_info[indices_tellu][self.key_filename].iloc[0])) as hdu:
                        mtrans = hdu['FLUX'].data
                else:
                    # use standard star spectrum as telluric model
                    mtrans = tellu_std

                indices_obj = indices & (self.product_info[self.key_target_name] == target)

                for file in self.product_info[indices_obj][self.key_filename]:
                    with fits.open(os.path.join(self.outpath, file)) as hdu:
                        specs = hdu["FLUX"].data
                        errs = hdu["FLUX_ERR"].data
                        # wave = hdu["WAVE"].data

                    f = np.ravel(specs)/mtrans/resp
                    f_err = np.ravel(errs)/mtrans/resp
                    # w = np.ravel(wave)

                    # mask deep telluric lines
                    mask = mtrans < 0.7
                    f[mask] = np.nan

                    file_name = os.path.join(self.outpath, file[:-5] + "1D_TELLURIC_CORR.dat")
                    np.savetxt(file_name, np.c_[w_std*1e-3, f, f_err], header="wave flux err")

                    print(f"Telluric corrected spectra saved to {file_name.split('/')[-1]}")

                    # plot spectra
                    specs = np.reshape(f, (3, -1, specs.shape[-1]))
                    waves = np.reshape(w_std, (3, -1, specs.shape[-1]))
                    self._plot_spec_by_order(file_name[:-4], specs, waves)


    def run_recipes(self, combine=False,
                    combine_mode='mean',
                    savename='',
                    companion_sep=None, 
                    remove_star_bkg=False, 
                    aper_prim=20, aper_comp=10,
                    extract_2d=True,
                    std_name=None,
                    std_prop=None,
                    run_molecfit=False, 
                    wave_range=None,
                    debug=False,
                    ):
        """
        Method for running the full chain of recipes.

        Parameters  
        ----------
        combine: bool
            If True, combine the individual frames at each nodding position.
        combine_mode: str
            The mode to use for combining the frames, either 'mean', 'median', or 'weighted'.
        savename: str
            The prefix of the output file names for companion sources on slit.
        companion_sep: float
            To extract spectra of the spatially resolved companion, 
            provide the separation of the companion from the primary 
            in arcsec.
        remove_star_bkg: bool
            If True, remove the background of the star in the companion extraction. 
            Note that this is experimental and should be used with caution.
        aper_prim: int
            The aperture size for the primary star extraction.
        aper_comp: int
            The aperture size for the companion extraction.
        extract_2d: bool
            If True, rectify the 2d image and save the intermediate 2d data product.
        std_name: str
            The name of the standard star. If `std_prop` is `None` the pipeline will attempt to automatically retrieve the star's properties in place of `std_prop`.
        std_prop: dict
            The properties of the standard star including the name, teff, vsini, and rv.
        run_molecfit: bool
            If True, run the molecfit for telluric correction.
        wave_range: list
            The wavelength range to use for the molecfit.
        debug: bool
            If True, show the debug plots.

        """
        
        if std_prop is None and std_name is not None:
            try:
                teff, vsini, rv = su.get_star_properties(std_name)
                std_prop = {'name': std_name, 'teff': teff, 'vsini': vsini, 'rv': rv}
                print(f"Retrieved star properties: {std_prop}")
            except ValueError as e:
                print("Unable to retrieve the star properties automatically. Please provide them in `std_prop`.")
                raise e

        
        self.extract_header()
        self.cal_dark()
        self.cal_flat_raw()
        self.cal_flat_trace()
        self.cal_slit_curve()
        self.cal_flat_norm()
        self.obs_nodding()

        if combine:
            self.obs_nodding_combine(
                                     combine_mode=combine_mode)
            input_type = f'{self.obs_mode}_COMBINED'
        else:
            input_type = f'{self.obs_mode}_FRAME'

        self.obs_extract(
                        caltype=input_type,
                        aper_prim=aper_prim,
                        aper_comp=aper_comp,
                        companion_sep=companion_sep, 
                        remove_star_bkg=remove_star_bkg,
                        extract_2d=extract_2d,
                        debug=debug,
                        savename=savename,
                        )
        
        self.refine_wlen_solution()

        if run_molecfit:
            self.run_molecfit(wave_range=wave_range)
        
        if std_prop is not None:
            self.spec_response_cal(object=std_prop['name'], 
                                   temp=std_prop['teff'], 
                                   vsini=std_prop['vsini'], 
                                   vsys=std_prop['rv'])
            self.apply_correction()


    def preprocessing(self, combine_mode='mean'):
        """
        Method for running the preprocessing recipes including reducing calibration files, 
        calibrating science frames, and combining indvidual frames at each nodding position.

        Parameters
        ----------
        combine_mode: str
            The mode to use for combining the frames.

        """
        self.extract_header()
        self.cal_dark()
        self.cal_flat_raw()
        self.cal_flat_trace()
        self.cal_slit_curve()
        self.cal_flat_norm()
        self.obs_nodding()

        self.obs_nodding_combine(combine_mode=combine_mode)

