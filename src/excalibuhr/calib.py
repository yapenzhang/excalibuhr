# File: src/excalibuhr/calib.py
__all__ = []


from ctypes import util
from fileinput import filename
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
# from numpy.polynomial import polynomial as Poly
# from scipy import ndimage
# from scipy.interpolate import CubicSpline, interp1d
import excalibuhr.utils as su
import excalibuhr.plotting as pu
import matplotlib.pyplot as plt 

class Pipeline:

    def __init__(self, workpath, night, clear=False, **header_keys):
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
        # self.detlin_path = '/run/media/yzhang/disk/cr2res_cal_detlin_coeffs.fits'

        if not os.path.exists(os.path.join(self.workpath, self.night)):
            os.makedirs(os.path.join(self.workpath, self.night))

        if not os.path.exists(self.calpath):
            os.makedirs(self.calpath)

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        if not os.path.exists(self.rawpath):
            os.makedirs(self.rawpath)

        # if clear:
        #     os.remove(self.header_file)
        #     os.remove(self.calib_file)

        if os.path.isfile(self.header_file):
            self.header_info = pd.read_csv(self.header_file, sep=';')
            print(self.header_info)
        else:
            self.header_info = None 

        for par in header_keys.keys():
            setattr(self, par, header_keys[par])

        if os.path.isfile(self.calib_file):
            self.calib_info = pd.read_csv(self.calib_file, sep=';')
            print(self.calib_info)
        else:
            self.calib_info = None

        if os.path.isfile(self.product_file):
            self.product_info = pd.read_csv(self.product_file, sep=';')
            print(self.product_info)
        else:
            self.product_info = None        


    def download_rawdata_eso(self, login, facility='eso', instrument='crires', **filters):

        print("Downloading rawdata from {}/{}...\n".format(facility, instrument))
        for key in filters.keys():
            print(key + ': '+ filters[key])

        if facility == 'eso':
            eso = Eso()
            eso.login(login)
            table = eso.query_instrument(instrument, column_filters={'night': self.night, **filters}) 
            data_files = eso.retrieve_data(table['DP.ID'], destination=self.rawpath, 
                        continuation=False, with_calib='raw', request_all_objects=True, unzip=False)
            os.chdir(self.rawpath)
            os.system("rm *.xml")
            os.system("rm *.txt")
            os.system("uncompress *.Z")
            os.chdir(self.workpath)


    def extract_header(self):

        print("Extracting FITS headers...\n")

        # keywords = ['ARCFILE', 'ORIGFILE', 'DATE-OBS', 'RA', 'DEC', 'OBJECT', 'MJD-OBS', \
        #             'ESO OBS TARG NAME', 'ESO OBS PROG ID', 'ESO OBS ID', 'ESO OBS WATERVAPOUR', \
        #             'ESO TPL ID', 'ESO DPR CATG', 'ESO DPR TECH', 'ESO DPR TYPE', 'ESO DET EXP ID', \
        #             'ESO DET SEQ1 DIT', 'ESO DET NDIT', 'ESO SEQ NEXPO', 'ESO SEQ NODPOS', 'ESO SEQ NODTHROW', \
        #             'ESO SEQ CUMOFFSETX', 'ESO SEQ CUMOFFSETY', 'ESO SEQ JITTERVAL', 'ESO TEL AIRM START', \
        #             'ESO TEL IA FWHM', 'ESO TEL AMBI TAU0', 'ESO TEL AMBI IWV START', 'ESO INS WLEN CWLEN', \
        #             'ESO INS GRAT1 ORDER', 'ESO INS WLEN ID', 'ESO INS SLIT1 NAME', 'ESO INS SLIT1 WID', \
        #             'ESO INS1 OPTI1 NAME', 'ESO INS1 DROT POSANG', 'ESO INS1 FSEL ALPHA', 'ESO INS1 FSEL DELTA', \
        #             'ESO AOS RTC LOOP STATE']
        keywords = self.header_keys.values() 
        
        raw_files = Path(self.rawpath).glob("*.fits")

        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:
            header = fits.getheader(file_item)
            if self.key_filename in header:
                shutil.move(file_item, os.path.join(self.rawpath, header[self.key_filename]))

            for key_item in keywords:
                if key_item in header:
                    header_dict[key_item].append(header[key_item])
                else:
                    header_dict[key_item].append(None)

        self.header_info = pd.DataFrame(data=header_dict)
        
        self.header_info.to_csv(os.path.join(self.calpath,'header_info.txt'), index=False, sep=';')

    def add_to_calib(self, file, cal_type):
        keywords = self.header_keys.values()
        header = fits.getheader(os.path.join(self.calpath, file))
        calib_dict = {}
        for key_item in keywords:
            calib_dict[key_item] = [header.get(key_item)]
        calib_dict[self.key_caltype] = [cal_type]
        calib_dict[self.key_filename] = [file]
        calib_append = pd.DataFrame(data=calib_dict)
        
        if self.calib_info is None:
            self.calib_info = calib_append
        else:
            self.calib_info = pd.concat([self.calib_info, calib_append], ignore_index=True)
        
        self.calib_info.to_csv(os.path.join(self.calpath,'calib_info.txt'), index=False, sep=';')

    def add_to_product(self, file, prod_type):
        keywords = self.header_keys.values()
        header = fits.getheader(os.path.join(self.outpath, file))
        calib_dict = {}
        for key_item in keywords:
            calib_dict[key_item] = [header.get(key_item)]
        calib_dict[self.key_caltype] = [prod_type]
        calib_dict[self.key_filename] = [file]
        calib_append = pd.DataFrame(data=calib_dict)
        
        if self.product_info is None:
            self.product_info = calib_append
        else:
            self.product_info = pd.concat([self.product_info, calib_append], ignore_index=True)
        
        self.product_info.to_csv(os.path.join(self.outpath,'product_info.txt'), index=False, sep=';')

    def cal_dark(self, verbose=True):
        indices = self.header_info[self.key_dtype] == "DARK"

        # Check unique DIT
        unique_dit = set()
        for item in self.header_info[indices][self.key_DIT]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # master dark
        for item in unique_dit:
            indices_dit = self.header_info[indices][self.key_DIT] == item
            dt = []
            for file in self.header_info[indices][indices_dit][self.key_filename]:
                with fits.open(os.path.join(self.rawpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))
            
            master, rons, badpix = su.util_master_dark(dt, badpix_clip=5)
            
            file_name = os.path.join(self.calpath, 'DARK_MASTER_DIT{}.fits'.format(item))
            su.wfits(file_name, master, hdr)
            self.add_to_calib('DARK_MASTER_DIT{}.fits'.format(item), "DARK_MASTER")
            
            file_name = os.path.join(self.calpath, 'DARK_RON_DIT{}.fits'.format(item))
            su.wfits(file_name, rons, hdr)
            self.add_to_calib('DARK_RON_DIT{}.fits'.format(item), "DARK_RON")

            file_name = os.path.join(self.calpath, 'DARK_BPM_DIT{}.fits'.format(item))
            su.wfits(file_name, badpix.astype(int), hdr)
            self.add_to_calib('DARK_BPM_DIT{}.fits'.format(item), "DARK_BPM")

            if verbose:
                print(r"{0:.1f}% of pixels identified as bad.".format(np.sum(badpix)/badpix.size*100.))
            # TODO: plot product
            # pu.plot_det_image()

    
    
    def cal_flat_raw(self, verbose=True):
        indices = self.header_info[self.key_dtype] == "FLAT"

        # Only utilize one DIT setting for master flat
        unique_dit = set()
        for item in self.header_info[indices][self.key_DIT]:
            unique_dit.add(item)
        dit = max(unique_dit)

        if len(unique_dit) > 1:
            print(f"Unique DIT values: {unique_dit}\n")
            print(f"Only utilize the flat with DIT of {dit}\n")
        elif len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") & (self.calib_info[self.key_DIT] == dit)
        indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") & (self.calib_info[self.key_DIT] == dit)
        assert (indices_dark.sum())<2
        for file in self.calib_info[indices_dark][self.key_filename]:
            dark = fits.getdata(os.path.join(self.calpath, file))
        for file in self.calib_info[indices_bpm][self.key_filename]:
            badpix = fits.getdata(os.path.join(self.calpath, file))

        # master flat
        for item_wlen in unique_wlen:
            indices_dit = indices & ((self.header_info[self.key_DIT] == dit) & (self.calib_info[self.key_wlen] == item_wlen))

            dt = []
            for file in self.header_info[indices_dit][self.key_filename]:
                with fits.open(os.path.join(self.rawpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))

            master, badpix = su.util_master_flat(dt, dark, collapse='median')

            file_name = os.path.join(self.calpath, 'FLAT_MASTER_w{}.fits'.format(item_wlen))
            su.wfits(file_name, master, hdr)
            self.add_to_calib('FLAT_MASTER_w{}.fits'.format(item_wlen), "FLAT_MASTER")

            file_name = os.path.join(self.calpath, 'FLAT_BPM_DIT{}.fits'.format(item))
            su.wfits(file_name, badpix.astype(int), hdr)
            self.add_to_calib('FLAT_BPM_DIT{}.fits'.format(item), "FLAT_BPM")

            if verbose:
                print(r"{0:.1f}% of pixels identified as bad.".format(np.sum(badpix)/badpix.size*100.))
            

    def cal_flat_trace(self, verbose=True, debug=False):
        indices = self.calib_info[self.key_caltype] == "FLAT_MASTER"

        # Check unique DIT
        unique_dit = set()
        for item in self.calib_info[indices][self.key_DIT]:
            unique_dit.add(item)
        dit = max(unique_dit)
        print(f"Use DIT value: {dit}\n")
        
        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.calib_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        # indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & (self.calib_info[self.key_DIT] == dit)
        # for file in self.calib_info[indices_bpm][self.key_filename]:
        #     bpm = fits.getdata(os.path.join(self.calpath, file))
        
        for item_wlen in unique_wlen:
            indices_flat = indices & ((self.calib_info[self.key_DIT] == dit) & (self.calib_info[self.key_wlen] == item_wlen))
            assert (indices_flat.sum())<2
            for file in self.calib_info[indices_flat][self.key_filename]:
                flat = fits.getdata(os.path.join(self.calpath, file))
                hdr = fits.getheader(os.path.join(self.calpath, file))
            
            trace = su.util_order_trace(flat, debug=debug)

            file_name = os.path.join(self.calpath, 'TW_FLAT_w{}.fits'.format(item_wlen))
            su.wfits(file_name, trace, hdr)
            self.add_to_calib('TW_FLAT_w{}.fits'.format(item_wlen), "TRACE_TW")
            

    def cal_slit_curve(self, debug=False):

        indices_fpet = self.header_info[self.key_dtype] == "WAVE,FPET"

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices_fpet][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        for item_wlen in unique_wlen:
            indices = self.header_info[indices_fpet][self.key_wlen] == item_wlen
            dit = self.header_info[indices_fpet][indices][self.key_DIT].iloc[0]
            file_fpet = self.header_info[indices_fpet][indices][self.key_filename].iloc[0]

            indices_dark = (self.calib_info[self.key_caltype] == "DARK_MASTER") & (self.calib_info[self.key_DIT] == dit)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM")
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & (self.calib_info[self.key_wlen] == item_wlen)
            assert (indices_dark.sum())<2
            assert (indices_bpm.sum())<2
            assert (indices_tw.sum())<2
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_dark][self.key_filename].iloc[0]
            dark = fits.getdata(os.path.join(self.calpath, file))
            for file in self.calib_info[indices_bpm][self.key_filename]:
                bpm = fits.getdata(os.path.join(self.calpath, file))

            with fits.open(os.path.join(self.rawpath, file_fpet)) as hdu:
                hdr = hdu[0].header
                fpet = np.array([hdu[i].data for i in range(1, len(hdu))]) - dark

            slit = su.util_slit_curve(fpet, bpm, tw, debug=debug)

            file_name = os.path.join(self.calpath, 'SLIT_TILT_w{}.fits'.format(item_wlen))
            su.wfits(file_name, slit, hdr)
            self.add_to_calib('SLIT_TILT_w{}.fits'.format(item_wlen), "SLIT_TILT")


    

    def cal_flat_norm(self, debug=True):
        indices = self.calib_info[self.key_caltype] == "FLAT_MASTER"

        # Check unique DIT
        unique_dit = set()
        for item in self.calib_info[indices][self.key_DIT]:
            unique_dit.add(item)
        dit = max(unique_dit)
        print(f"Use DIT value: {dit}\n")
        
        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.calib_info[indices][self.key_wlen]:
            unique_wlen.add(item)

        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

        for item_wlen in unique_wlen:
            indices_flat = indices & ((self.calib_info[self.key_DIT] == dit) & (self.calib_info[self.key_wlen] == item_wlen))

            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & (self.calib_info[self.key_DIT] == dit)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") & (self.calib_info[self.key_wlen] == item_wlen)
            assert (indices_bpm.sum())<2
            assert (indices_tw.sum())<2
            assert (indices_slit.sum())<2
            file = self.calib_info[indices_tw][self.key_filename].iloc[0]
            tw = fits.getdata(os.path.join(self.calpath, file))
            file = self.calib_info[indices_slit][self.key_filename].iloc[0]
            slit = fits.getdata(os.path.join(self.calpath, file))
            for file in self.calib_info[indices_bpm][self.key_filename]:
                bpm = fits.getdata(os.path.join(self.calpath, file))

            assert (indices_flat.sum())<2
            for file in self.calib_info[indices_flat][self.key_filename]:
                flat = fits.getdata(os.path.join(self.calpath, file))
                hdr = fits.getheader(os.path.join(self.calpath, file))
            
            flat_norm, blazes = su.util_master_flat_norm(flat, bpm, tw, slit, debug=debug)

            file_name = os.path.join(self.calpath, 'FLAT_NORM_w{}.fits'.format(item_wlen))
            su.wfits(file_name, flat_norm, hdr)
            self.add_to_calib('FLAT_NORM_w{}.fits'.format(item_wlen), "FLAT_NORM")

            file_name = os.path.join(self.calpath, 'BLAZE_w{}.fits'.format(item_wlen))
            su.wfits(file_name, blazes, hdr)
            self.add_to_calib('BLAZE_w{}.fits'.format(item_wlen), "BLAZE")


    def obs_nodding(self, combine=False, debug=True):
        self.noddingpath = os.path.join(self.outpath, "obs_nodding")
        if not os.path.exists(self.noddingpath):
            os.makedirs(self.noddingpath)

        indices = self.header_info[self.key_catg] == "SCIENCE"
        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.header_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        if len(unique_wlen) == 0:
            print("Unique WLEN settings: none")
        else:
            print(f"Unique WLEN settings: {unique_wlen}\n")

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

            indices_flat = (self.calib_info[self.key_caltype] == "FLAT_NORM") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & (self.calib_info[self.key_wlen] == item_wlen)
            
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
            
            for item_dit in unique_dit:

                indices_ron = (self.calib_info[self.key_caltype] == "DARK_RON") & (self.calib_info[self.key_DIT] == item_dit)
                file_ron = os.path.join(self.calpath, self.calib_info[indices_ron][self.key_filename].iloc[0])
                ron = fits.getdata(os.path.join(self.calpath, file_ron))

                indices_nod_A = indices_wlen & (self.header_info[self.key_DIT] == item_dit) & (self.header_info[self.key_nodpos] == 'A')
                indices_nod_B = indices_wlen & (self.header_info[self.key_DIT] == item_dit) & (self.header_info[self.key_nodpos] == 'B')
                df_nods = self.header_info[indices_nod_A | indices_nod_B].sort_values(self.key_filename)

                nod_a_count = sum(indices_nod_A)
                nod_b_count = sum(indices_nod_B)
                Nexp_per_nod = int(nod_a_count//self.header_info[indices_nod_A][self.key_nabcycle].iloc[0])

                print(f"Number of exposures at nod A: {nod_a_count}")
                print(f"Number of exposures at nod B: {nod_b_count}")

                assert nod_a_count == nod_b_count, "There is an unequal number of exposures at nod A and nod B."
                # if nod_a_count != nod_b_count:
                #     warnings.warn("There is an unequal number of exposures at nod A and nod B.")
                # print(df_nods)
                
                # nodding pair AB subtraction
                nodthrow = df_nods[self.key_nodthrow].iloc[0]
                slitlen = df_nods[self.key_slitlen].iloc[0]
                
                for i, row in enumerate(range(0, df_nods.shape[0], Nexp_per_nod)):
                    pos_bkg = set()
                    for p in df_nods[self.key_nodpos].iloc[row:row+Nexp_per_nod]:
                        pos_bkg.add(p)
                    print("BKG: ", pos_bkg)
                    file_list = [os.path.join(self.rawpath, item) for item in df_nods[self.key_filename].iloc[row:row+Nexp_per_nod]]
                    dt_list, err_list = [], []
                    for file in file_list:
                        frame, frame_err = [], []
                        with fits.open(file) as hdu:
                            ndit = hdu[0].header[self.key_NDIT]
                            for j,d in enumerate(range(1, len(hdu))):
                                gain = hdu[d].header[self.key_gain]
                                frame.append(hdu[d].data)
                                frame_err.append(su.detector_shotnoise(hdu[d].data, ron[j], GAIN=gain, NDIT=ndit))
                        dt_list.append(frame)
                        err_list.append(frame_err)
                    dt_bkg, err_bkg = su.combine_detector_images(dt_list, err_list, collapse='mean')
                    
                    pos = set()
                    for p in df_nods[self.key_nodpos].iloc[row+(-1)**(i%2)*Nexp_per_nod:row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]:
                        pos.add(p)
                    assert pos != pos_bkg, "Subtracting frames at the same nodding position."
                    for file in df_nods[self.key_filename].iloc[row+(-1)**(i%2)*Nexp_per_nod:row+(-1)**(i%2)*Nexp_per_nod+Nexp_per_nod]:
                        print("files",file, "POS: ", pos)
                        frame, frame_err = [], []
                        with fits.open(os.path.join(self.rawpath, file)) as hdu:
                            hdr = hdu[0].header
                            ndit = hdr[self.key_NDIT]
                            for j,d in enumerate(range(1, len(hdu))):
                                gain = hdu[d].header[self.key_gain]
                                frame.append(hdu[d].data)
                                frame_err.append(su.detector_shotnoise(hdu[d].data, ron[j], GAIN=gain, NDIT=ndit))
                        frame_bkg_cor, err_bkg_cor = su.combine_detector_images([frame, -dt_bkg], [frame_err, err_bkg], collapse='sum')
                        frame_bkg_cor, err_bkg_cor = su.util_correct_readout_artifact(frame_bkg_cor, err_bkg_cor, bpm, tw, debug=False)
                        frame_bkg_cor, err_bkg_cor = su.util_flat_fielding(frame_bkg_cor, err_bkg_cor, flat, debug=False)

                        file_name = os.path.join(self.noddingpath, 'Nodding_{}'.format(file))
                        su.wfits(file_name, frame_bkg_cor, hdr, im_err=err_bkg_cor)
                        self.add_to_product("./obs_nodding/"+'Nodding_{}'.format(file), "NODDING_FRAME")
                        # plt.imshow((frame_bkg_cor)[2], vmin=-20, vmax=20)
                        # plt.show()
                        # plt.imshow((frame_bkg_cor/flat)[2], vmin=-20, vmax=20)
                        # plt.show()
                        if hdr[self.key_nodpos]=='A':
                            nod = -1
                        else:
                            nod = 1
                        # if not combine:
                        #   extract
                
                if combine:
                    for pos in ['A', 'B']:
                        dt, dt_err = [], []
                        indices = (self.product_info[self.key_caltype] == 'NODDING_FRAME') & (self.product_info[self.key_nodpos] == pos)
                        for file in self.product_info[indices][self.key_filename]:
                            with fits.open(os.path.join(self.outpath, file)) as hdu:
                                hdr = hdu[0].header
                                dt.append(hdu[0].data)
                                dt_err.append(hdu[1].data)
                        combined, combined_err = su.combine_detector_images(dt, dt_err, collapse='mean')
                        file_name = os.path.join(self.noddingpath, 'Nodding_combined_{}_{}.fits'.format(pos, item_wlen))
                        su.wfits(file_name, combined, hdr, im_err=combined_err)
                        self.add_to_product("./obs_nodding/"+'Nodding_combined_{}_{}.fits'.format(pos, item_wlen), "NODDING_COMBINED")

    def extract1d_from_combined(self, companion_sep=None, debug=True):        
        indices = (self.product_info[self.key_caltype] == 'NODDING_COMBINED')

        # Check unique WLEN setting
        unique_wlen = set()
        for item in self.product_info[indices][self.key_wlen]:
            unique_wlen.add(item)
        print(f"Unique WLEN settings: {unique_wlen}\n")

        for item_wlen in unique_wlen:
            
            indices_wlen = indices & (self.product_info[self.key_wlen] == item_wlen)

            indices_blaze = (self.calib_info[self.key_caltype] == "BLAZE") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_tw = (self.calib_info[self.key_caltype] == "TRACE_TW") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_slit = (self.calib_info[self.key_caltype] == "SLIT_TILT") & (self.calib_info[self.key_wlen] == item_wlen)
            indices_bpm = (self.calib_info[self.key_caltype] == "FLAT_BPM") & (self.calib_info[self.key_wlen] == item_wlen)
            
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

            dt, dt_err = [], []
            for file in self.product_info[indices_wlen][self.key_filename]:
                with fits.open(os.path.join(self.outpath, file)) as hdu:
                    hdr = hdu[0].header
                    dt = hdu[0].data
                    dt_err = hdu[1].data
                if hdr[self.key_nodpos]=='A':
                    nod = -1
                else:
                    nod = 1                    
                nodthrow = hdr[self.key_nodthrow]
                slitlen = hdr[self.key_slitlen]
                f0 = 0.5+nod*nodthrow/2./slitlen
                if companion_sep is None:
                    print("Location of target on slit: ", f0)
                    su.util_extract_spec(dt, dt_err, bpm, tw, slit, blaze, f0=f0, debug=debug)
                else:
                    f1 = f0 - companion_sep/slitlen
                    print("Location of star and companion: {0:.3f}, {1:.3f}".format(f0, f1))
                    su.util_extract_spec(dt, dt_err, bpm, tw, slit, blaze, f0=f1, aper_half=10, debug=debug)
                    # su.util_extract_spec(dt, dt_err, bpm, tw, slit, blaze, f0=f0, debug=debug)


        # savename = outpath / f"spec_{tag[:3]}_nod{which_nod}_{j:02d}.npz"
        # np.savez(savename, WAVE=wlen, FLUX=spec, FLUX_ERR=spec_err)

        # stdout = subprocess.DEVNULL #None
        # subprocess.run(esorex, cwd=outpath, stdout=stdout, check=True)



