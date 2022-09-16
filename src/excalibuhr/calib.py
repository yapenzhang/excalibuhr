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
from astropy.stats import sigma_clip
from astroquery.eso import Eso
from numpy.polynomial import polynomial as Poly
from scipy import ndimage
from scipy.interpolate import CubicSpline, interp1d
import excalibuhr.utils as su
import excalibuhr.plotting as pu
import matplotlib.pyplot as plt 

class Pipeline:

    def __init__(self, workpath, night, clear=False, wave_method='fpet', **header_keys):
        self.workpath = os.path.abspath(workpath)
        self.night = night
        self.key_caltype = 'CAL TYPE'
        self.header_keys = header_keys
        self.rawpath = os.path.join(self.workpath, self.night, "raw")
        self.calpath = os.path.join(self.workpath, self.night, "cal")
        self.outpath = os.path.join(self.workpath, self.night, "out")
        self.calib_file = os.path.join(self.calpath, "calib_info.txt") 
        self.header_file = os.path.join(self.calpath, "header_info.txt")
        # self.detlin_path = '/run/media/yzhang/disk/cr2res_cal_detlin_coeffs.fits'
        # self.wave_method = wave_method
        # self.NAXIS = 2048
        # self.RON = 6
        # self.GAIN = 2.1
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
            calib_dict[key_item] = [header[key_item]]
        calib_dict[self.key_caltype] = [cal_type]
        calib_dict[self.key_filename] = [file]
        calib_append = pd.DataFrame(data=calib_dict)
        
        if self.calib_info is None:
            self.calib_info = calib_append
        else:
            self.calib_info = pd.concat([self.calib_info, calib_append], ignore_index=True)
        
        self.calib_info.to_csv(os.path.join(self.calpath,'calib_info.txt'), index=False, sep=';')


    def cal_dark(self, badpix_clip=4, verbose=True):
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
            master = np.nanmedian(dt, axis=0)
            badpix = np.zeros_like(master).astype(bool)
            for i, det in enumerate(master):
                filtered_data = sigma_clip(det, sigma=badpix_clip, axis=0)
                badpix[i] = filtered_data.mask
                master[i][badpix[i]] = np.nan
            
            file_name = os.path.join(self.calpath, 'DARK_MASTER_DIT{}.fits'.format(item))
            su.wfits(file_name, master, hdr)
            self.add_to_calib('DARK_MASTER_DIT{}.fits'.format(item), "DARK_MASTER")
            
            file_name = os.path.join(self.calpath, 'DARK_BPM_DIT{}.fits'.format(item))
            su.wfits(file_name, badpix.astype(int), hdr)
            self.add_to_calib('DARK_BPM_DIT{}.fits'.format(item), "DARK_BPM")

            if verbose:
                print(r"{0:.1f}% of pixels identified as bad.".format(np.sum(badpix)/badpix.size*100.))
            # TODO: plot product
            # pu.plot_det_image()

    
    
    def cal_flat_raw(self, verbose=True, collapse='median'):
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
            if collapse=='median':
                master = np.nanmedian(dt, axis=0)
            else:
                master = np.nanmean(dt, axis=0)

            master -= dark

            # TODO: apply bpm; plot

            file_name = os.path.join(self.calpath, 'FLAT_MASTER_w{}.fits'.format(item_wlen))
            su.wfits(file_name, master, hdr)
            self.add_to_calib('FLAT_MASTER_w{}.fits'.format(item_wlen), "FLAT_MASTER")
            

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

        indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") & (self.calib_info[self.key_DIT] == dit)
        for file in self.calib_info[indices_bpm][self.key_filename]:
            bpm = fits.getdata(os.path.join(self.calpath, file))
        
        for item_wlen in unique_wlen:
            indices_flat = indices & ((self.calib_info[self.key_DIT] == dit) & (self.calib_info[self.key_wlen] == item_wlen))
            assert (indices_flat.sum())<2
            for file in self.calib_info[indices_flat][self.key_filename]:
                flat = fits.getdata(os.path.join(self.calpath, file))
                hdr = fits.getheader(os.path.join(self.calpath, file))
            trace = su.util_order_trace(flat, bpm, debug=debug)

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
            indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") & (self.calib_info[self.key_DIT] == dit)
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

            indices_bpm = (self.calib_info[self.key_caltype] == "DARK_BPM") & (self.calib_info[self.key_DIT] == dit)
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
            blaze = su.util_extract_blaze(flat, bpm, tw, slit, debug=debug)

            # file_name = os.path.join(self.calpath, 'TW_FLAT_w{}.fits'.format(item_wlen))
            # su.wfits(file_name, trace, hdr)
            # self.add_to_calib('TW_FLAT_w{}.fits'.format(item_wlen), "TRACE_TW")



    def obs_nodding(self, STD=False, verbose=False):
        if STD:        
            indices = self.header_data["DPR.TYPE"] == "STD"
            outpath = self.stdpath

            # Check unique DIT

            unique_dit = set()
            for item in self.header_data[indices]["DET.SEQ1.DIT"]:
                unique_dit.add(item)

            print(f"Unique DIT values: {unique_dit}\n")

            # Count nod positions

            nod_a_exp = (self.header_data["SEQ.NODPOS"] == "A") & \
                        (self.header_data["DPR.TYPE"] == "STD")

            nod_b_exp = (self.header_data["SEQ.NODPOS"] == "B") & \
                        (self.header_data["DPR.TYPE"] == "STD")

            nod_a_count = sum(nod_a_exp)
            nod_b_count = sum(nod_b_exp)

            print(f"Number of exposures at nod A: {nod_a_count}")
            print(f"Number of exposures at nod B: {nod_b_count}")

            if nod_a_count != nod_b_count:
                warnings.warn("There is an unequal number of exposures "
                            "at nod A and nod B.")
        else:
            indices = self.header_data["DPR.CATG"] == "SCIENCE"
            outpath = self.objpath

            # Check unique DIT

            unique_dit = set()
            for item in self.header_data[indices]["DET.SEQ1.DIT"]:
                unique_dit.add(item)

            print(f"Unique DIT values: {unique_dit}\n")

            # Count nod positions

            nod_a_exp = (self.header_data["SEQ.NODPOS"] == "A") & \
                        (self.header_data["DPR.CATG"] == "SCIENCE")

            nod_b_exp = (self.header_data["SEQ.NODPOS"] == "B") & \
                        (self.header_data["DPR.CATG"] == "SCIENCE")

            nod_a_count = sum(nod_a_exp)
            nod_b_count = sum(nod_b_exp)

            print(f"Number of exposures at nod A: {nod_a_count}")
            print(f"Number of exposures at nod B: {nod_b_count}")

            if nod_a_count != nod_b_count:
                warnings.warn("There is an unequal number of exposures "
                            "at nod A and nod B.")

        # Create SOF file
        nods = self.header_data[indices].sort_values(by=['ARCFILE'])

        sof_file = pathlib.Path(outpath / "nodd.sof")
        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in nods["ARCFILE"]:
                sof_open.write(f"{self.rawpath}/{item} OBS_NODDING_OTHER\n")

            file_found = False
            if "CAL_FLAT_MASTER" in self.calibDB:
                for key, value in self.calibDB["CAL_FLAT_MASTER"].items():
                    if not file_found and value["WLEN"] == self.header_data[indices].iloc[0]["INS.WLEN.ID"]:
                        sof_open.write(f"{key} CAL_FLAT_MASTER\n")
                        file_found = True

            file_found = False
            if "CAL_DARK_BPM" in self.calibDB:
                for key, value in self.calibDB["CAL_DARK_BPM"].items():
                    if not file_found and value["DIT"] == self.header_data[indices].iloc[0]["DET.SEQ1.DIT"]:
                        sof_open.write(f"{key} CAL_DARK_BPM\n")
                        file_found = True
            
            file_found = False
            if "CAL_WAVE_TW" in self.calibDB:
                for key, value in self.calibDB["CAL_WAVE_TW"].items():
                    if not file_found and key[-8:] == self.wave_method+'.fits':
                        sof_open.write(f"{key} UTIL_WAVE_TW\n")
                        file_found = True
            if not file_found:
                raise RuntimeError("TW TABLE file not found.")

            sof_open.write(f"{self.detlin_path} CAL_DETLIN_COEFFS\n")

        # Run EsoRex
        esorex = [
            "esorex",
            # f"--recipe-config={config_file}",
            f"--output-dir={outpath}",
            "cr2res_obs_nodding",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=outpath, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        if STD:
            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_combinedA.fits")
            for item in fits_files:
                self.add_to_calib_DB("STD_NODDING_COMBINEDA", str(item))

        else:
            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_combinedA.fits")
            for item in fits_files:
                self.add_to_calib_DB("OBS_NODDING_COMBINEDA", str(item))


        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.calibDB, json_file, indent=4)

    # def _RectifyTraceInt(self, data, curve, peak=None):
    #     """
    #     Shift 2D data, row by row, according to curve
    #     ----------

    #     Parameters
    #     ----------
    #     curve : the shift at each row
    #     peak : pivot curve 

    #     Returns
    #     ----------
    #     rectified image
    #     """
    #     # def simple_shift(xs, ns):
    #     # mask = np.isnan(data)
    #     # data = data*(~mask).astype(float)
    #     if peak == None:
    #         # peak = np.argmax(np.nansum(data, axis=1))
    #         peak = len(curve)//2
    #     # pivot curve around peak and change sign so shift is corrective 
    #     curve_p  = -1.0 * (curve - curve[peak])
    #     curve_p = np.rint(curve_p).astype(int)
    #     e = np.empty_like(data)
    #     for i, n in enumerate(curve_p):
    #         if n == 0:
    #             e[i] = data[i]
    #         elif n > 0:
    #             e[i,:n] = np.nan
    #             e[i,n:] = data[i,:-n]
    #         else:
    #             e[i,n:] = np.nan
    #             e[i,:n] = data[i,-n:]
    #     return e

    def _RectifyTraceInt(self, data, curve, peak=None):
        """
        Shift 2D data, row by row, according to curve
        ----------

        Parameters
        ----------
        curve : the shift at each row
        peak : pivot curve 

        Returns
        ----------
        rectified image
        """
        mask = np.isnan(data)
        data = data*(~mask).astype(float)
        if peak == None:
            # peak = np.argmax(np.nansum(data, axis=1))
            peak = len(curve)//2
        # pivot curve around peak and change sign so shift is corrective 
        curve_p  = -1.0 * (curve - curve[peak])
        rectified, mask_shift = [], []
        for i in range(len(curve_p)):
            s = data[i]
            # if np.all(np.isnan(s)):
            #     rectified.append(np.ones_like(s)*np.nan)
            # else:
                # s = interp1d(np.arange(len(s))[~np.isnan(s)], s[~np.isnan(s)], bounds_error=False, fill_value=0)(np.arange(len(s)))
            rectified.append(ndimage.interpolation.shift(
                    s, curve_p[i], order=1, mode='nearest', prefilter=True)) 
            mask_shift.append(ndimage.interpolation.shift(
                    mask[i].astype(float), curve_p[i], order=0, mode='nearest', prefilter=True)) 
        rectified = np.array(rectified)
        rectified[np.array(mask_shift) > 0] = np.nan
        return(rectified)

    def simple_optimal_combine(self, D, V, M_bp_init, clip_sigma=3,  plotting=False):
        wave_x = np.arange(D.shape[0])
        spatial_x = np.arange(D.shape[1])
        f_std = np.sum(D*M_bp_init, axis=1)
        D_norm = np.zeros_like(D)
        P = np.zeros_like(D)
        for x in spatial_x:
            D_norm[:,x] = D[:,x]/(f_std+1e-9)
        for x in spatial_x:
            poly_P = PolyfitClip(wave_x, \
                            D_norm[:,x], 3, \
                            clip=4, plotting=False)
            P[:,x] = Poly.polyval(wave_x, poly_P)
        P[P<=0] = 1e-9
        for w in wave_x:
            P[w] /= np.sum(P[w])
        D_norm = D/P
        D_med = np.median(D_norm*M_bp_init, axis=1)
        V_norm = V/P**2

        res_norm = np.abs(D_norm - np.tile(D_med, (D_norm.shape[1],1)).T)
        res = res_norm/np.sqrt(V_norm)
        M_bp = res < clip_sigma

        M_bp = M_bp*M_bp_init

        V_norm += res_norm**2
        f_opt = np.sum(M_bp*P*D, axis=1)/(np.sum(M_bp*P*P, axis=1)+1e-9)
        var = np.sum(M_bp*P, axis=1)/(np.sum(M_bp/V_norm, axis=1)+1e-9)

        # interpolate over bad wavelength channels
        bad_channel = np.sum(~M_bp, axis=1)>(M_bp.shape[1]/2.-1.)
        interp_flux = interp1d(np.arange(len(f_opt))[~bad_channel], f_opt[~bad_channel], bounds_error=False, fill_value='extrapolate')
        # interp_var = interp1d(np.arange(len(var))[~bad_channel], var[~bad_channel], bounds_error=False, fill_value='extrapolate')
        f_opt = interp_flux(np.arange(len(f_opt)))
        var[bad_channel] = 1e6

        if plotting:
            plt.plot(f_std)
            plt.plot(f_opt)
            # plt.plot(bad_channel)
            plt.show()
            plt.plot(f_opt/np.sqrt(var))
            plt.show()
        return f_opt, var, P, M_bp

    def OptimalExtraction(self, D, V, width_obj, width_sky=None, RON=6, GAIN=2, returnprofile=False):

        wave_x = np.arange(D.shape[0])
        spatial_x = np.arange(D.shape[1])
        obj_x = np.argmax(np.nanmedian(np.nan_to_num(D), axis=0))

        if width_sky:
            # mask obj before fitting the sky
            mask_obj = np.ones(D.shape[1], dtype=bool)
            mask_obj[obj_x-width_sky:obj_x+width_sky+1]=False

            S = np.zeros_like(D)
            for w in wave_x:
                poly_sky = PolyfitClip(spatial_x[mask_obj], \
                                D[w,mask_obj], 1, \
                                # w=1./V[w,mask_obj], \
                                clip=4, plotting=False)
                S[w] = Poly.polyval(spatial_x, poly_sky)
            D_sub = (D-S)[:,obj_x-width_obj:obj_x+width_obj]
            V_sub = V[:,obj_x-width_obj:obj_x+width_obj]
            # plt.plot(D[1000])
            # plt.show()
        else:
            D_sub = D[:,obj_x-width_obj:obj_x+width_obj]
            V_sub = V[:,obj_x-width_obj:obj_x+width_obj]

        # nan mask
        #badpixel mask in D_sub
        M_nan = np.isnan(D_sub)
        count_level = np.median(np.sort(D[:,obj_x][~np.isnan(D[:,obj_x])])[-100:])
        bp_mask = M_nan|(D_sub>count_level*2.)|(D_sub<-0.1)
        # plt.imshow(D_sub, vmin=-20, vmax=20)
        # plt.imshow(bp_mask)
        # plt.show()
        D_sub = np.nan_to_num(D_sub, nan=1e-9)
        V_sub = np.nan_to_num(V_sub, nan=1e-9)

        f_opt, var, P, M_bp = self.simple_optimal_combine(D_sub, V_sub, ~bp_mask)
        if returnprofile:
            return f_opt, var, P, M_bp
        else:
            return f_opt, var

    def extract_spec(self, STD=False, orders=[4,3], extract_width=10):
        detctors = [1,2,3]
        x = np.arange(1, self.NAXIS+1)

        if STD:
            tag = "STD_NODDING_COMBINED"
            outpath = self.stdpath
        else:
            tag = "OBS_NODDING_COMBINED"
            outpath = self.objpath
        
        if "CAL_WAVE_TW" in self.calibDB:
            for key, value in self.calibDB["CAL_WAVE_TW"].items():
                if key[-8:] == self.wave_method+'.fits':
                    tw_filename = key

        for which_nod in ['A', 'B']:
            if tag+which_nod in self.calibDB:
                for j, obs_filename in enumerate(self.calibDB[tag+which_nod].keys()):
                    wlen, spec, spec_err = [],[],[]
                    for o, order in enumerate(orders):
                        for e, ext in enumerate(detctors):
                            dt =  fits.getdata(obs_filename, 2*ext-1)
                            dt_err =  fits.getdata(obs_filename, 2*ext)

                            tw = fits.getdata(tw_filename, ext)
                            p_upper = tw['Upper'][tw['Order']==order][0]
                            y_upper = Poly.polyval(x, p_upper) 
                            p_lower = tw['Lower'][tw['Order']==order][0]
                            y_lower = Poly.polyval(x, p_lower) 
                            p_wave = tw['Wavelength'][tw['Order']==order][0]
                            wave = Poly.polyval(x, p_wave)

                            im = dt[int(y_lower.min()-1.):int(y_upper.max()-1.)]
                            im_err = dt_err[int(y_lower.min()-1.):int(y_upper.max()-1.)]

                            """trace rectify"""
                            p_trace = tw['All'][tw['Order']==order][0]
                            y_trace = Poly.polyval(x, p_trace) - y_lower.min()
                            # plt.imshow(im[:,:], vmin=0, vmax=500)
                            # # plt.plot(x-1., y_trace-y_trace[0], '-r')
                            # plt.show()
                            im_rect_trace = (self._RectifyTraceInt(im.T, y_trace)).T
                            im_rect_trace_err = (self._RectifyTraceInt(im_err.T, y_trace)).T
                            # plt.imshow(im_rect_trace[:,:], vmin=0, vmax=400)
                            # plt.show()

                            """spectral rectify"""
                            slitAa = tw['SlitPolyA'][tw['Order']==order][0][0] 
                            slitAb = tw['SlitPolyA'][tw['Order']==order][0][1] 
                            slitBa = tw['SlitPolyB'][tw['Order']==order][0][0]
                            slitBb = tw['SlitPolyB'][tw['Order']==order][0][1]
                            slitA = Poly.polyval(x, [slitAa, slitAb])
                            slitB = Poly.polyval(x, [slitBa, slitBb])
                            isowlen_grid = []
                            for (pA, pB) in zip(slitA, slitB):
                                isowlen_grid.append(Poly.polyval(np.arange(y_lower.min(), y_upper.max()), [pA, pB]))
                            isowlen_grid = np.array(isowlen_grid).T
                            im_rect_spec = np.copy(im_rect_trace)
                            im_rect_err = np.copy(im_rect_trace_err)
                            
                            for i, (x_isowlen, data_row, err_row) in enumerate(zip(isowlen_grid, im_rect_trace, im_rect_trace_err)):
                                mask = np.isnan(data_row)
                                if np.sum(mask)>0.5*len(mask):
                                    continue
                                im_rect_spec[i] = interp1d(x[~mask], data_row[~mask], bounds_error=False, fill_value=np.nan)(x_isowlen)
                                im_rect_err[i] = interp1d(x[~mask], err_row[~mask], bounds_error=False, fill_value=np.nan)(x_isowlen)

                            # plt.imshow(im, vmin=-20, vmax=20)
                            # plt.show()
                            # plt.imshow(im_rect_spec, vmin=-20, vmax=20)
                            # plt.show()

                            obj_pos = np.argmax(np.nanmedian(np.nan_to_num(im_rect_spec), axis=1))
                            if obj_pos<len(im_rect_spec)//2:
                                im_rect_spec = im_rect_spec[:len(im_rect_spec)//2]
                                im_rect_err = im_rect_err[:len(im_rect_err)//2]
                            else:
                                im_rect_spec = im_rect_spec[len(im_rect_spec)//2:]
                                im_rect_err = im_rect_err[len(im_rect_err)//2:]

                            f_opt, var = self.OptimalExtraction(im_rect_spec.T, im_rect_err.T**2, extract_width, int(2*extract_width), self.RON, self.GAIN)
                            wlen.append(wave)
                            spec.append(f_opt)
                            spec_err.append(var)
                    savename = outpath / f"spec_{tag[:3]}_nod{which_nod}_{j:02d}.npz"
                    np.savez(savename, WAVE=wlen, FLUX=spec, FLUX_ERR=spec_err)
                    self.add_to_calib_DB(f"spec_{tag[:3]}_nod{which_nod}", str(savename))
            else:
                raise RuntimeError(f"{tag} file not found.")
        
        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.calibDB, json_file, indent=4)

    def check_extracted(self, tag=['spec_OBS_nodA']):
        cmap = plt.get_cmap("tab10")
        for i, nod in enumerate(tag):
            for key in self.calibDB[nod].keys():
                dt = np.load(key)
                wlen, spec, var = dt['WAVE'], dt['FLUX'], dt['FLUX_ERR']
                for x,y,y_err in zip(wlen, spec, var):
                    plt.plot(x, y, color=cmap(i))
        plt.show()

