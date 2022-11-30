# File: src/excalibuhr/calib.py
__all__ = []


import os
import sys
import json
import shutil
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import subprocess
from astropy.io import fits
from astroquery.eso import Eso
import skycalc_ipy
import excalibuhr.utils as su
import excalibuhr.plotting as pu

class Pipeline:

    def __init__(self, workpath: str, night: str, clean_start: bool = False,
                 **header_keys: dict) -> None:
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
        clean_start: bool
            Set it to True to remove the infomation files when redoing the 
            entire reduction.
        header_keys: dict
            The needed keyword names in the header of input ``.fits`` files.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            f"Run pipeline for Night: {night}", bound_char="=")
        
        self.workpath = os.path.abspath(workpath)
        self.night = night
        self.key_caltype = 'CAL TYPE'
        self.header_keys = header_keys
        self.rawpath = os.path.join(self.workpath, self.night, "raw")
        self.calpath = os.path.join(self.workpath, self.night, "cal")
        self.outpath = os.path.join(self.workpath, self.night, "out")
        self.calib_file = os.path.join(self.calpath, "calib_info.txt") 
        self.header_file = os.path.join(self.calpath, "header_info.txt")
        self.product_file = os.path.join(self.outpath, "product_info.txt")
        self.gain=[2.15, 2.19, 2.0]
        self.pix_scale=0.056 #arcsec
        
        print("Data reduction folder:"
              f"{os.path.join(self.workpath, self.night)}")

        # self.detlin_path = '/run/media/yzhang/disk/cr2res_cal_detlin_coeffs.fits'

        # Create the directories if they do not exist
        if not os.path.exists(os.path.join(self.workpath, self.night)):
            os.makedirs(os.path.join(self.workpath, self.night))

        if not os.path.exists(self.calpath):
            os.makedirs(self.calpath)

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        if not os.path.exists(self.rawpath):
            os.makedirs(self.rawpath)

        # in case redo the entire reduction 
        if clean_start:
            os.remove(self.header_file)
            os.remove(self.calib_file)
            os.remove(self.prodct_file)

        # If present, read the info files
        if os.path.isfile(self.header_file):
            print("Reading header data from header_info.txt")
            self.header_info = pd.read_csv(self.header_file, sep=';')
        else:
            self.header_info = None 

        for par in header_keys.keys():
            setattr(self, par, header_keys[par])

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


    def _print_section(self, sect_title: str, bound_char: str = "-", 
                        extra_line: bool = True) -> None:
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

        Returns
        -------
        NoneType
            None
        """

        if extra_line:
            print("\n" + len(sect_title) * bound_char)
        else:
            print(len(sect_title) * bound_char)

        print(sect_title)
        print(len(sect_title) * bound_char + "\n")


    def download_rawdata_eso(self, login: str, facility: str = 'eso', 
                instrument: str ='crires', **filters) -> None:
        """
        Method for downloading raw data from eso archive using astroquery

        Parameters
        ----------
        login : str
            username to login to the ESO User Portal Services
        filters: 
            optional parameters for data filtering, e.g. prog_id

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Downloading rawdata from ESO arhive")

        print('Night: '+ self.night)
        for key in filters.keys():
            print(key + ': '+ filters[key])

        if facility == 'eso':
            eso = Eso()
            eso.login(login)
            table = eso.query_instrument(
                instrument, column_filters={'night': self.night, **filters}
                ) 
            data_files = eso.retrieve_data(
                table['DP.ID'], destination=self.rawpath, 
                continuation=False, with_calib='raw', 
                request_all_objects=True, unzip=False)
            os.chdir(self.rawpath)
            os.system("rm *.xml")
            os.system("rm *.txt")
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
        ``DataFrame`` and a text file.

        Returns
        -------
        NoneType
            None
        """
        #if self.header_info is not None:
        #    # Header extraction was already performed
        #    return

        self._print_section("Extracting Observation details")

        print("Extracting header details to `header_info.txt`")
        keywords = self.header_keys.values() 
        
        raw_files = Path(self.rawpath).glob("*.fits")

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
        self.header_info.to_csv(
            os.path.join(self.calpath,'header_info.txt'), index=False, sep=';')

    def _add_to_calib(self, file: str, cal_type: str) -> None:
        """
        Internal method for adding details of processed calibration files
        to the DataFrame and text file.

        Parameters
        ----------
        file : str
            filename to be added to the table
        cal_type: str 
            type of the calibration file, e.g. `DARK_MASTER`, `FLAT_MASTER`

        Returns
        -------
        NoneType
            None
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
        
        self.calib_info.to_csv(os.path.join(self.calpath,'calib_info.txt'), 
                                    index=False, sep=';')

    def _add_to_product(self, file: str, prod_type: str) -> None:
        """
        Internal method for adding details of data products
        to the DataFrame and text file.

        Parameters
        ----------
        file : str
            filename to be added to the table
        cal_type: str 
            type of the data product, e.g. `NODDING_FRAME`, `Extr1D_PRIMARY`

        Returns
        -------
        NoneType
            None
        """
        print(f"{prod_type}: out/{file}")
        header = fits.getheader(os.path.join(self.outpath, file))

        calib_dict = {}
        keywords = self.header_keys.values()
        for key_item in keywords:
            calib_dict[key_item] = [header.get(key_item)]
        calib_dict[self.key_caltype] = [prod_type]
        calib_dict[self.key_filename] = [file]
        calib_append = pd.DataFrame(data=calib_dict)
        
        if self.product_info is None:
            self.product_info = calib_append
        else:
            self.product_info = pd.concat([self.product_info, calib_append], 
                                        ignore_index=True)
        
        self.product_info.to_csv(os.path.join(self.outpath,'product_info.txt'), 
                                        index=False, sep=';')

    def cal_dark(self, clip: int = 5, collapse: str = 'median') -> None:
        """
        Method for combining dark frames according to DIT.

        Parameters
        ----------
        clip : int
            sigma of bad pixel clipping
        collapse: str
            the way of combining dark frames: `mean` or `median`

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create DARK_MASTER")
        
        indices = (self.header_info[self.key_dtype] == "DARK")

        # Check unique DIT
        unique_dit = set()
        for item in self.header_info[indices][self.key_DIT]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            raise RuntimeError("No DARK frames found in the raw folder")
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
                                    collapse=collapse)
            
            print("\n Output files:")
            # Save the master dark, read-out noise, and bad-pixel maps
            file_name = os.path.join(self.calpath, 
                            f'DARK_MASTER_DIT{item}.fits')
            su.wfits(file_name, master, hdr)
            self._add_to_calib(f'DARK_MASTER_DIT{item}.fits', 
                            "DARK_MASTER")
            pu.plot_det_image(master, file_name, 
                            f"DARK_MASTER, DIT={item:.1f}")
            
            file_name = os.path.join(self.calpath, 
                            f'DARK_RON_DIT{item}.fits')
            su.wfits(file_name, rons, hdr)
            self._add_to_calib(f'DARK_RON_DIT{item}.fits', 
                            "DARK_RON")

            file_name = os.path.join(self.calpath, 
                            f'DARK_BPM_DIT{item}.fits')
            su.wfits(file_name, badpix.astype(int), hdr)
            self._add_to_calib(f'DARK_BPM_DIT{item}.fits', 
                            "DARK_BPM")

            print(f"DIT {item:.1f} s -> "
                  f"{np.sum(badpix)/badpix.size*100.:.1f}"
                  r"% of pixels identified as bad")

    
    def cal_flat_raw(self, clip: int = 5, collapse: str = 'median') -> None:
        """
        Method for combining raw flat frames according to wavelngth setting.

        Parameters
        ----------
        clip : int
            sigma of bad pixel clipping
        collapse: str
            the way of combining multiple frames: `mean` or `median`

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create FLAT_MASTER")

        indices = self.header_info[self.key_dtype] == "FLAT"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            raise RuntimeError("No FLAT frames found in the raw folder")
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
            if len(unique_dit) == 0:
                raise RuntimeError("No FLAT frames found in the raw folder")
            dit = max(unique_dit)
            
            indices_dit = indices_wlen & (self.header_info[self.key_DIT] == dit) 

            # Select master dark and bad-pixel mask corresponding to DIT
            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") \
                         & (self.calib_info[self.key_DIT] == dit)
            indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") \
                        & (self.calib_info[self.key_DIT] == dit)
            if np.sum(indices_dark) < 1:
                raise RuntimeError(f"No MASTER DARK frame found with the \
                        DIT value {dit}s corresponding to that of FLAT frames")

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

            # Per detector, median-combine the flats and determine the bad pixels
            master, badpix = su.util_master_flat(dt, dark, 
                            badpix_clip=clip, collapse=collapse)
            
            print(f"WLEN setting {item_wlen} -> " 
                  f"{np.sum(badpix)/badpix.size*100.:.1f}"
                  r"% of pixels identified as bad")

            print("\n Output files:")
            # Save the master flat and bad-pixel map
            file_name = os.path.join(self.calpath, \
                            f'FLAT_MASTER_{item_wlen}.fits')
            su.wfits(file_name, master, hdr)
            self._add_to_calib(f'FLAT_MASTER_{item_wlen}.fits', "FLAT_MASTER")

            file_name = os.path.join(self.calpath, 
                            f'FLAT_BPM_{item_wlen}.fits')
            su.wfits(file_name, badpix.astype(int), hdr)
            self._add_to_calib(f'FLAT_BPM_{item_wlen}.fits', 
                            "FLAT_BPM")
            

    def cal_flat_trace(self, sub_factor: int = 64) -> None:
        """
        Method for identifying traces of spectral order in `MASTER FLAT`.

        Parameters
        ----------
        sub_factor : int
            binning factor along the dispersion axis.

        Returns
        -------
        NoneType
            None
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
            trace = su.util_order_trace(flat, bpm, 
                    hdr[self.key_slitlen]/self.pix_scale, sub_factor=sub_factor)
            
            print("\n Output files:")
            # Save the polynomial coefficients
            file_name = os.path.join(self.calpath, f'TW_FLAT_{item_wlen}.fits')
            su.wfits(file_name, trace, hdr)
            self._add_to_calib(f'TW_FLAT_{item_wlen}.fits', "TRACE_TW")
            
            pu.plot_det_image(flat, file_name, f"FLAT_MASTER_{item_wlen}", 
                            tw=trace)
            

    def cal_slit_curve(self, debug=False) -> None:
        """
        Method for tracing slit curvature in the `FPET` calibration frame.
        Determine initial wavelength solution by mapping FPET lines to 
        an evenly spaced wavelength grid.

        Parameters
        ----------
        debug : bool
            generate plots for debugging.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Trace slit curvature")

        # Select the Fabry-Perot etalon calibrations
        indices_fpet = self.header_info[self.key_dtype] == "WAVE,FPET"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices_fpet][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            raise RuntimeError("No FPET frame found in the raw folder")

        # Identify the slit curvature for each WLEN setting
        for item_wlen in unique_wlen:
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
                raise RuntimeError(f"No MASTER DARK frame found with the \
                        DIT value {dit}s corresponding to that of FPET frame")             
            if np.sum(indices_tw) < 1:
                raise RuntimeError(f"No order trace (TRACE_TW) found with \
                        the WLEN setting {item_wlen} corresponding to  \
                        that of FPET frame") 

            # Read the trace-wave, master dark and bad-pixel mask
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))

            file = self.calib_info[indices_dark][self.key_filename].iloc[0]
            dark = fits.getdata(os.path.join(self.calpath, file))

            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))

            wlen_mins, wlen_maxs = [], []
            with fits.open(os.path.join(self.rawpath, file_fpet)) as hdu:
                # Dark-subtract the fpet observation
                hdr = hdu[0].header
                fpet = np.array([hdu[i].data for i in range(1, len(hdu))]) \
                        - dark
                
                # Store the minimum and maximum wavelengths
                # of each order {j} in each detector {i}.
                for i in range(1, len(hdu)):
                    wlen_min, wlen_max = [], []
                    header = hdu[i].header
                    for j in range(1,11): # maximum 10 orders possible
                        if float(header[self.key_wave_cen+str(j)]) > 0:
                            wlen_min.append(header[self.key_wave_min+str(j)])
                            wlen_max.append(header[self.key_wave_max+str(j)])
                    wlen_mins.append(wlen_min)
                    wlen_maxs.append(wlen_max)

            # Assess the slit curvature and wavelengths along the orders
            slit, wlens = su.util_slit_curve(fpet, bpm, tw, 
                            wlen_mins, wlen_maxs, debug=debug)

            print("\n Output files:")
            # Save the polynomial coefficients describing the slit curvature 
            # and an initial wavelength solution
            file_name = os.path.join(self.calpath, 
                            f'SLIT_TILT_{item_wlen}.fits')
            su.wfits(file_name, slit, hdr)
            self._add_to_calib(f'SLIT_TILT_{item_wlen}.fits', "SLIT_TILT")

            pu.plot_det_image(fpet, file_name, f"FPET_{item_wlen}", 
                            tw=tw, slit=slit)

            file_name = os.path.join(self.calpath, 
                            f'INIT_WLEN_{item_wlen}.fits')
            su.wfits(file_name, wlens, hdr)
            self._add_to_calib(f'INIT_WLEN_{item_wlen}.fits', "INIT_WLEN")
            

    def cal_flat_norm(self, debug=True):
        """
        Method for creating normalized flat field and 
        extracting blaze function

        Parameters
        ----------
        debug : bool
            generate plots for debugging.

        Returns
        -------
        NoneType
            None
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
            flat_norm, blazes = su.util_master_flat_norm(flat, bpm, 
                                    tw, slit, debug=debug)

            print("\n Output files:")
            file_name = os.path.join(self.calpath, 
                                    f'FLAT_NORM_{item_wlen}.fits')
            su.wfits(file_name, flat_norm, hdr)
            self._add_to_calib(f'FLAT_NORM_{item_wlen}.fits', "FLAT_NORM")

            pu.plot_det_image(flat_norm, file_name, f"FLAT_NORM_{item_wlen}")

            file_name = os.path.join(self.calpath, f'BLAZE_{item_wlen}.fits')
            su.wfits(file_name, blazes, hdr)
            self._add_to_calib(f'BLAZE_{item_wlen}.fits', "BLAZE")
            

    def obs_nodding(self, debug=False):

        # Create the obs_nodding directory if it does not exist yet
        self.noddingpath = os.path.join(self.outpath, "obs_nodding")
        if not os.path.exists(self.noddingpath):
            os.makedirs(self.noddingpath)

        # Select the science observations
        indices = self.header_info[self.key_catg] == "SCIENCE"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        # Loop over each WLEN setting
        for item_wlen in unique_wlen:
            
            indices_wlen = indices & (self.header_info[self.key_wlen] == item_wlen)
            # Check unique DIT
            unique_dit = set()
            for item in self.header_info[indices_wlen][self.key_DIT]:
                unique_dit.add(item)
            if len(unique_dit) == 0:
                print("Wavelength setting {}; None DIT values found \n".format(item_wlen))
            else:
                print("Wavelength setting {}; Unique DIT values {} \n".format(item_wlen, unique_dit))

            # Select the corresponding calibration files
            indices_flat = (self.calib_info[self.key_caltype] == "FLAT_NORM") & \
                           (self.calib_info[self.key_wlen] == item_wlen)
            indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") & \
                            (self.calib_info[self.key_wlen] == item_wlen)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & \
                         (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") & \
                           (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & \
                          (self.calib_info[self.key_wlen] == item_wlen)
            
            assert (indices_flat.sum())<2, "More than one calibration file."
            assert (indices_blaze.sum())<2, "More than one calibration file."
            assert (indices_tw.sum())<2, "More than one calibration file."
            assert (indices_slit.sum())<2, "More than one calibration file."
            assert (indices_bpm.sum())<2, "More than one calibration file."

            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_slit][self.key_filename].iloc[0]
            slit = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_blaze][self.key_filename].iloc[0]
            blaze = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_flat][self.key_filename].iloc[0]
            flat = fits.getdata(os.path.join(self.calpath, file))
            
            # Loop over each DIT
            for item_dit in unique_dit:

                # Open the read-out noise file
                indices_ron = (self.calib_info[self.key_caltype] == "DARK_RON")# & \
                              #(self.calib_info[self.key_DIT] == item_dit)
                file_ron = os.path.join(self.calpath, self.calib_info[indices_ron][self.key_filename].iloc[0])
                ron = fits.getdata(os.path.join(self.calpath, file_ron))

                indices_nod_A = indices_wlen & (self.header_info[self.key_DIT] == item_dit) & \
                                (self.header_info[self.key_nodpos] == 'A')
                indices_nod_B = indices_wlen & (self.header_info[self.key_DIT] == item_dit) & \
                                (self.header_info[self.key_nodpos] == 'B')
                df_nods = self.header_info[indices_nod_A | indices_nod_B].sort_values(self.key_filename)

                nod_a_count = sum(indices_nod_A)
                nod_b_count = sum(indices_nod_B)
                #print(nod_a_count, self.header_info[indices_nod_A][self.key_nabcycle])
                #print(self.key_nabcycle)
                #Nexp_per_nod = int(nod_a_count//self.header_info[indices_nod_A][self.key_nabcycle].iloc[0])
                Nexp_per_nod = int(self.header_info[indices_nod_A][self.key_nexp_per_nod].iloc[0])

                print(f"Number of exposures at nod A: {nod_a_count}")
                print(f"Number of exposures at nod B: {nod_b_count}")

                assert nod_a_count == nod_b_count, "There is an unequal number of exposures at nod A and nod B."
                # if nod_a_count != nod_b_count:
                #     warnings.warn("There is an unequal number of exposures at nod A and nod B.")
                # print(df_nods)
                
                for i, row in enumerate(range(0, df_nods.shape[0], Nexp_per_nod)):
                    # Select the following background measurement 
                    # (i.e. the other nod position)
                    pos_bkg = set()
                    for p in df_nods[self.key_nodpos].iloc[row:row+Nexp_per_nod]:
                        pos_bkg.add(p)
                    print("BKG: ", pos_bkg)
                    
                    file_list = [os.path.join(self.rawpath, item) \
                                 for item in df_nods[self.key_filename].iloc[row:row+Nexp_per_nod]
                                 ]
                    dt_list, err_list = [], []
                    # Loop over the observations at the next nod position
                    for file in file_list:
                        frame, frame_err = [], []
                        with fits.open(file) as hdu:
                            ndit = hdu[0].header[self.key_NDIT]
                            
                            # Loop over the detectors
                            for j, d in enumerate(range(1, len(hdu))):
                                # gain = hdu[d].header[self.key_gain]
                                frame.append(hdu[d].data)
                                # Calculate the shot-noise for this detector
                                frame_err.append(su.detector_shotnoise(hdu[d].data, ron[j], GAIN=self.gain[j], NDIT=ndit))
                                
                        dt_list.append(frame)
                        err_list.append(frame_err)
                        
                    # Mean-combine the images if there are multiple exposures per nod
                    dt_bkg, err_bkg = su.combine_frames(dt_list, err_list, collapse='mean')
                    
                    # Select the current nod position
                    pos = set()
                    for p in df_nods[self.key_nodpos].iloc[row+(-1)**(i%2)*Nexp_per_nod:\
                                                           row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]:
                        pos.add(p)
                    
                    assert pos != pos_bkg, "Subtracting frames at the same nodding position."

                    # Loop over the observations of the current nod position
                    for file in df_nods[self.key_filename].iloc[row+(-1)**(i%2)*Nexp_per_nod:
                                                                row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]:
                        print("files", file, "POS: ", pos)
                        frame, frame_err = [], []
                        with fits.open(os.path.join(self.rawpath, file)) as hdu:
                            hdr = hdu[0].header
                            ndit = hdr[self.key_NDIT]

                            # Loop over the detectors
                            for j, d in enumerate(range(1, len(hdu))):
                                frame.append(hdu[d].data)
                                # Calculate the shot-noise for this detector
                                frame_err.append(np.zeros_like(hdu[d].data))
                                # frame_err.append(su.detector_shotnoise(hdu[d].data, ron[j], GAIN=gain, NDIT=ndit))
                        
                        # Subtract the nod-pair from each other
                        frame_bkg_cor, err_bkg_cor = su.combine_frames([frame, -dt_bkg], [frame_err, err_bkg], collapse='sum')
                        # correct vertical strips due to readout artifacts
                        frame_bkg_cor, err_bkg_cor = su.util_correct_readout_artifact(frame_bkg_cor, err_bkg_cor, bpm, tw, debug=False)
                        # Apply the flat-fielding
                        frame_bkg_cor, err_bkg_cor = su.util_flat_fielding(frame_bkg_cor, err_bkg_cor, flat, debug=False)


                        file_name = os.path.join(self.noddingpath, 'Nodding_{}'.format(file))
                        su.wfits(file_name, frame_bkg_cor, hdr, ext_list=err_bkg_cor)
                        self._add_to_product("./obs_nodding/"+'Nodding_{}'.format(file), "NODDING_FRAME")
                        # plt.imshow((frame_bkg_cor/flat)[2], vmin=-20, vmax=20)
                        # plt.show()
                        if hdr[self.key_nodpos] == 'A':
                            nod = -1
                        else:
                            nod = 1
                        # TODO: time series observations
                        # if not combine:
                        #   extract here
    
    def obs_nodding_combine(self, debug=True):

        # Create the obs_nodding directory if it does not exist yet
        self.noddingpath = os.path.join(self.outpath, "obs_nodding")
        if not os.path.exists(self.noddingpath):
            os.makedirs(self.noddingpath)
        
        # Select the obs_nodding observations
        indices = (self.product_info[self.key_caltype] == 'NODDING_FRAME')

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.product_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        # Loop over each WLEN setting
        for item_wlen in unique_wlen:            
            indices_wlen = indices & (self.product_info[self.key_wlen] == item_wlen)

            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & \
                         (self.calib_info[self.key_wlen] == item_wlen)

            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))

            # Loop over the nodding positions
            for pos in ['A', 'B']:
                indices = indices_wlen & (self.product_info[self.key_nodpos] == pos)
                frames, frames_err = [], []

                # Loop over the observations at each nodding position
                for j, file in enumerate(self.product_info[indices][self.key_filename]):
                    with fits.open(os.path.join(self.outpath, file)) as hdu:
                        hdr = hdu[0].header
                        # in case of jittering
                        if np.isclose(hdr[self.key_jitter], 0):
                            dt, dt_err = hdu[0].data, hdu[1].data
                        else:
                            # integer shift only!
                            dt, dt_err = su.align_jitter(hdu[0].data, hdu[1].data, int(np.round(hdr[self.key_jitter]/self.pix_scale)), tw, debug=False)
                        
                        frames.append(dt)
                        frames_err.append(dt_err)

                        # plt.imshow(hdu[1].data[0], vmin=0, vmax=8)
                        # plt.show()

                # Mean-combine the images per detector for each nodding position
                print("Combining {0:d} frames at slit position {1:s}...".format(j+1, pos))
                combined, combined_err = su.combine_frames(frames, frames_err, collapse='mean')
                # plt.imshow(combined_err[0]<1)
                # plt.imshow(combined_err[0], vmin=0, vmax=8)
                # plt.show()

                # Save the combined obs_nodding observation
                file_name = os.path.join(self.noddingpath, 'Nodding_combined_{}_{}.fits'.format(pos, item_wlen))
                su.wfits(file_name, combined, hdr, ext_list=combined_err)
                self._add_to_product("./obs_nodding/"+'Nodding_combined_{}_{}.fits'.format(pos, item_wlen), "NODDING_COMBINED")

    def extract1d_nodding(self, f_star=None, companion_sep=None, 
                          aper_prim=20, aper_comp=10, debug=False):    
        
        # Create the obs_nodding directory if it does not exist yet
        self.noddingpath = os.path.join(self.outpath, "obs_nodding")
        if not os.path.exists(self.noddingpath):
            os.makedirs(self.noddingpath)

        # Select the combined obs_nodding observations
        indices = (self.product_info[self.key_caltype] == 'NODDING_COMBINED')

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.product_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        print(f"Unique WLEN settings: {unique_wlen}\n")

        # Loop over each WLEN setting
        for item_wlen in unique_wlen:
            
            indices_wlen = indices & (self.product_info[self.key_wlen] == item_wlen)

            # Select the corresponding calibration files
            indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") & \
                            (self.calib_info[self.key_wlen] == item_wlen)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & \
                         (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") & \
                           (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & \
                          (self.calib_info[self.key_wlen] == item_wlen)
            # indices_wave = (self.calib_info[self.key_caltype] == "INIT_WLEN") & \
            #                (self.calib_info[self.key_wlen] == item_wlen)
            
            assert (indices_blaze.sum())<2, "More than one calibration file."
            assert (indices_tw.sum())<2, "More than one calibration file."
            assert (indices_slit.sum())<2, "More than one calibration file."
            assert (indices_bpm.sum())<2, "More than one calibration file."

            file = self.calib_info[indices_bpm][self.key_filename].iloc[0]
            bpm = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_slit][self.key_filename].iloc[0]
            slit = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_blaze][self.key_filename].iloc[0]
            blaze = fits.getdata(os.path.join(self.calpath, file))
            # file = self.calib_info[indices_wave][self.key_filename].iloc[0]
            # wlens = fits.getdata(os.path.join(self.calpath, file))
            
            # Loop over each combined obs_nodding observation
            for file in self.product_info[indices_wlen][self.key_filename]:

                with fits.open(os.path.join(self.outpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt = hdu[0].data
                    dt_err = hdu[1].data
                pos = hdr[self.key_nodpos]
                slitlen = hdr[self.key_slitlen]
                # if pos == 'A':
                #     nod = -1
                # else:
                #     nod = 1                    
                # nodthrow = hdr[self.key_nodthrow]
                if f_star is None:
                    # determine the location of peak signal from data
                    f0 = su.peak_slit_fraction(dt[0], tw[0], debug=debug) 
                else:
                    f0 = f_star[pos]
                    # # Slit-fraction of centered for the nod-throw
                    # f = 0.5 + nod*nodthrow/2./slitlen

                # The slit is centered on the target
                print("Location of target on slit: ", f0)
                
                # # Extract a 1D spectrum for the target
                flux_pri, err_pri = su.util_extract_spec(dt, dt_err, bpm, tw, slit, blaze,  
                                                         gains=self.gain, f0=f0, 
                                                         aper_half=aper_prim, debug=False)

                file_name = os.path.join(self.noddingpath, 'Extr1D_Nodding_combined_{}_{}_{}.fits'.format(pos, item_wlen, 'PRIMARY'))
                su.wfits(file_name, flux_pri, hdr, ext_list=err_pri)
                self._add_to_product("./obs_nodding/"+'Extr1D_Nodding_combined_{}_{}_{}.fits'.format(pos, item_wlen, 'PRIMARY'), "Extr1D_PRIMARY")
                
                if not companion_sep is None:
                    # The slit is centered on the star, not the companion
                    f1 = f0 - companion_sep/slitlen
                    print("Location of star and companion: {0:.3f}, {1:.3f}".format(f0, f1))

                    # Extract a 1D spectrum for the secondary
                    flux_sec, err_sec = su.util_extract_spec(dt, dt_err, bpm, tw, slit, blaze, 
                                                             gains=self.gain, f0=f0, f1=f1, 
                                                             aper_half=aper_comp, bkg_subtract=True,
                                                             f_star=flux_pri,
                                                             debug=debug)

                    file_name = os.path.join(self.noddingpath, 'Extr1D_Nodding_combined_{}_{}_{}.fits'.format(pos, item_wlen, 'SECONDARY'))
                    su.wfits(file_name, flux_sec, hdr, ext_list=err_sec)
                    self._add_to_product("./obs_nodding/"+'Extr1D_Nodding_combined_{}_{}_{}.fits'.format(pos, item_wlen, 'SECONDARY'), "Extr1D_SECONDARY")
                


    def refine_wlen_solution(self,  mode='quad', debug=False):
        """
        Method for refining wavelength solution by maximizing 
        cross-correlation functions between the spectrum and a
        telluric transmission model on a order-by-order basis.

        Parameters
        ----------
        mode: str
            If mode is `linear`, then the wavelength solution is 
            corrected with a linear function. If mode is `quad`,
            the correction is a quadratic function.
        debug : bool
            generate plots for debugging.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Refine wavelength solution")


        self.noddingpath = os.path.join(self.outpath, "obs_nodding")
        if not os.path.exists(self.noddingpath):
            os.makedirs(self.noddingpath)

        indices = (self.product_info[self.key_caltype] == 'Extr1D_PRIMARY')

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.product_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        indices_tellu = (self.calib_info[self.key_caltype] == "TELLU") 
        if np.sum(indices_tellu) < 1:
            raise RuntimeError("No Telluric transmission model found. \
                        Please genrate it with `run_skycalc` first.") 
        file = self.calib_info[indices_tellu][self.key_filename].iloc[0]
        tellu = fits.getdata(os.path.join(self.calpath, file))

        for item_wlen in unique_wlen:
            print(f"Calibrating WLEN setting {item_wlen}:")
            
            indices_wlen = indices & \
                        (self.product_info[self.key_wlen] == item_wlen)

            indices_wave = (self.calib_info[self.key_caltype] == "INIT_WLEN") \
                         & (self.calib_info[self.key_wlen] == item_wlen)
            file = self.calib_info[indices_wave][self.key_filename].iloc[0]
            wlen_init = fits.getdata(os.path.join(self.calpath, file))

            indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") \
                          & (self.calib_info[self.key_wlen] == item_wlen)
            file = self.calib_info[indices_blaze][self.key_filename].iloc[0]
            blaze = fits.getdata(os.path.join(self.calpath, file))

            dt = []
            # sum spectra at A and B nodding positions
            for file in self.product_info[indices_wlen][self.key_filename]:
                with fits.open(os.path.join(self.outpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt.append(hdu[0].data)
            dt = np.sum(dt, axis=0)

            wlen_cal = su.util_wlen_solution(dt, wlen_init, blaze, tellu, 
                            mode=mode, debug=debug)
            
            print("\n Output files:")
            file_name = os.path.join(self.noddingpath,
                                    f'CAL_WLEN_{item_wlen}.fits')
            su.wfits(file_name, wlen_cal, hdr)
            self._add_to_product("./obs_nodding/"+f'CAL_WLEN_{item_wlen}.fits', 
                                    "CAL_WLEN")

    def save_extracted_data(self, debug=False):
        self.noddingpath = os.path.join(self.outpath, "obs_nodding")

        for target in ['PRIMARY', 'SECONDARY', 'STD']:
            indices = (self.product_info[self.key_caltype] == f'Extr1D_{target}')
            if np.sum(indices)>0:

                # Check unique WLEN setting
                unique_wlen = set()
                for item in self.product_info[indices][self.key_wlen]:
                    unique_wlen.add(item)
                print(f"Unique WLEN settings: {unique_wlen}\n")

                wlens, blazes, specs, errs = [],[],[],[]
                for item_wlen in unique_wlen:
                    
                    indices_wlen = indices & (self.product_info[self.key_wlen] == item_wlen)

                    indices_wave = (self.product_info[self.key_caltype] == "CAL_WLEN") & (self.product_info[self.key_wlen] == item_wlen)
                    file = self.product_info[indices_wave][self.key_filename].iloc[0]
                    wlen = fits.getdata(os.path.join(self.outpath, file))
                    wlens.append(wlen)

                    indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") & (self.calib_info[self.key_wlen] == item_wlen)
                    file = self.calib_info[indices_blaze][self.key_filename].iloc[0]
                    blaze = fits.getdata(os.path.join(self.calpath, file))
                    blazes.append(blaze)

                    dt, dt_err = [], []
                    for file in self.product_info[indices_wlen][self.key_filename]:
                        with fits.open(os.path.join(self.outpath, file)) as hdu:
                            # hdr = hdu[0].header
                            dt.append(hdu[0].data)
                            dt_err.append(hdu[1].data)
                    master, master_err = su.combine_frames(dt, dt_err, collapse='mean')
                    specs.append(master)
                    errs.append(master_err)

                file_name = os.path.join(self.outpath, f'SPEC_{target}.fits')
                su.wfits(file_name, specs, ext_list=[errs, wlens])

                #unravel spectra in each order and detector to a 1D array.
                w, f, f_err, w_even = su.util_unravel_spec(np.array(wlens), np.array(specs), np.array(errs), np.array(blazes), debug=debug)

                file_name = os.path.join(self.outpath, f'SPEC_{target}.dat')
                header = "Wlen(nm) Flux Flux_err Wlen_even"
                np.savetxt(file_name, np.c_[w, f, f_err, w_even], header=header)




    def run_skycalc(self, pwv: float = 5) -> None:
        """
        Method for running the Python wrapper of SkyCalc
        (see https://skycalc-ipy.readthedocs.io).

        Parameters
        ----------
        pwv : float
            Precipitable water vapor (default: 5) that is used for
            the telluric spectrum. 

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Run SkyCalc")

        # Indices with SCIENCE frames
        indices = self.header_info[self.key_catg] == "SCIENCE"

        # Setup SkyCalc object
        print("SkyCalc settings:")

        sky_calc = skycalc_ipy.SkyCalc()

        wlen_id= self.header_info[indices][self.key_wlen].iloc[0]
        slit_width= self.header_info[indices][self.key_slitwid].iloc[0]
        mjd_start = self.header_info[indices][self.key_mjd].iloc[0]
        ra_mean = np.mean(self.header_info[self.key_ra][indices])
        dec_mean = np.mean(self.header_info[self.key_dec][indices])

        sky_calc.get_almanac_data(
            ra=ra_mean,
            dec=dec_mean,
            date=None,
            mjd=mjd_start,
            observatory="paranal",
            update_values=True,
        )

        print(f"  - MJD = {mjd_start:.2f}")
        print(f"  - RA (deg) = {ra_mean:.2f}")
        print(f"  - Dec (deg) = {dec_mean:.2f}")

        # See https://skycalc-ipy.readthedocs.io/en/latest/GettingStarted.html
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
        sky_calc["wres"] = 2e5
        sky_calc["pwv"] = pwv

        print(f"  - Wavelength range (nm) = {sky_calc['wmin']} - {sky_calc['wmax']}")
        print(f"  - lambda / Dlambda = {sky_calc['wres']}")
        print(f"  - Airmass = {sky_calc['airmass']:.2f}")
        print(f"  - PWV (mm) = {sky_calc['pwv']}\n")

        # Get telluric spectra from SkyCalc

        print("Get telluric spectrum with SkyCalc...", end="", flush=True)

        # temp_file =  os.path.join(self.calpath , "skycalc_temp.fits")
        # sky_spec = sky_calc.get_sky_spectrum(filename=temp_file)
        wave, trans, _ = sky_calc.get_sky_spectrum(return_type="arrays")

        print(" [DONE]\n")

        # Convolve spectra

        if slit_width == "w_0.2":
            spec_res = 100000.0
        elif slit_width == "w_0.4":
            spec_res = 50000.0
        else:
            raise ValueError(f"Slit width {slit_width} not recognized.")

        print(f"Slit width = {slit_width}")
        print(f"Smoothing spectrum to R = {spec_res}\n")

        trans = su.SpecConvolve(wave.value, trans, out_res=spec_res, in_res=sky_calc["wres"])

        transm_spec = np.column_stack((1e3 * wave.value, trans))
        out_file= os.path.join(self.calpath, "TRANSM_SPEC.fits")
        su.wfits(out_file, transm_spec)
        self._add_to_calib('TRANSM_SPEC.fits', "TELLU")
        # header = "Wavelength (nm) - Transmission"
        # np.savetxt(out_file, transm_spec, header=header)

        # stdout = subprocess.DEVNULL #None
        # subprocess.run(esorex, cwd=outpath, stdout=stdout, check=True)

def CombineNights(workpath, night_list, collapse='weighted'):
    from scipy.interpolate import interp1d

    datapath = os.path.join(workpath, 'DATA')
    if not os.path.exists(datapath):
            os.makedirs(datapath)

    
    for target in ['SECONDARY', 'PRIMARY', 'STD']:
        specs = [] #, [], [], []
        for night in night_list:
            filename = os.path.join(workpath, night, 'out', f'SPEC_{target}.dat')
            if os.path.isfile(filename):
                specs.append(np.genfromtxt(filename, skip_header=1))
        if len(specs)<1:
            continue
        specs = np.array(specs)

        # interpolate to the common wavelength grid
        wlen_grid = specs[-1,:,3]
        wlens = specs[:,:,0]
        fluxes = specs[:,:,1]
        errs = specs[:,:,2]
        fluxes_grid = np.zeros_like(fluxes)
        errs_grid = np.zeros_like(errs)
        for i in range(specs.shape[0]):
            mask = np.isnan(fluxes[i]) | np.isnan(errs[i])
            fluxes_grid[i] = interp1d(wlens[i][~mask], fluxes[i][~mask], kind='cubic', 
                                      bounds_error=False, fill_value=np.nan
                                      )(wlen_grid)
            errs_grid[i] = interp1d(wlens[i][~mask], errs[i][~mask], kind='cubic', 
                                      bounds_error=False, fill_value=np.nan
                                      )(wlen_grid)

        # combine spectra
        master, master_err = su.combine_frames(fluxes_grid, errs_grid, collapse=collapse)

        file_name = os.path.join(datapath, f'SPEC_{target}.dat')
        header = "Wlen(nm) Flux Flux_err"
        np.savetxt(file_name, np.c_[wlen_grid, master, master_err], header=header)

        file_name = os.path.join(datapath, f'SPEC_{target}')
        pu.plot_spec1d(wlen_grid, master, master_err, file_name)