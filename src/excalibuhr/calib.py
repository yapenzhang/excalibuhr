# File: src/excalibuhr/calib.py
__all__ = []


import os
import sys
import json
import warnings
import pathlib
import numpy as np
import pandas as pd
import subprocess
from astropy.io import fits
from astropy.stats import sigma_clip
from astroquery.eso import Eso
from numpy.polynomial import polynomial as Poly
from scipy import ndimage
from scipy.interpolate import CubicSpline, interp1d
import utils as su
import matplotlib.pyplot as plt 


class Pipeline:

    def __init__(self, workpath, night, clear=False, wave_method='fpet', **header_keys):
        self.workpath = pathlib.Path(workpath).resolve()
        self.night = night
        self.header_keys = header_keys
        self.rawpath = pathlib.Path(self.workpath / self.night / "RAW")
        self.calpath = pathlib.Path(self.workpath / self.night / "CAL")
        self.outpath = pathlib.Path(self.workpath / self.night / "OUT")
        # self.objpath = pathlib.Path(self.path / self.night / "obj")
        # self.stdpath = pathlib.Path(self.path / self.night / "std")
        self.calib_file = self.calpath / "calib_info.txt"
        self.header_file = self.calpath / "header_info.txt"
        # self.detlin_path = '/run/media/yzhang/disk/cr2res_cal_detlin_coeffs.fits'
        # self.wave_method = wave_method
        # self.NAXIS = 2048
        # self.RON = 6
        # self.GAIN = 2.1
        if not os.path.exists(self.workpath/self.night):
            os.makedirs(self.workpath/self.night)

        if not os.path.exists(self.calpath):
            os.makedirs(self.calpath)

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        if not os.path.exists(self.rawpath):
            os.makedirs(self.rawpath)

        if clear:
            os.remove(self.header_file)
            os.remove(self.calib_file)

        if self.header_file.is_file():
            self.header_info = pd.read_csv(self.header_file, sep=' ')
            print(self.header_info)
        else:
            self.header_info = pd.DataFrame()

        column_name = ["CAL.TYPE"]
        for par in header_keys.keys():
            setattr(self, par, header_keys[par])
            column_name.append(header_keys[par])

        if self.calib_file.is_file():
            self.calib_info = pd.read_csv(self.calib_file, sep=' ')
            print(self.calib_info)
        else:
            self.calib_info = pd.DataFrame(columns=column_name)

        


    def download_rawdata(self, facility='eso', instrument='crires', login='yzhang', **filters):

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
            os.system("uncompress *.Z")
            os.chdir(self.workpath)


    def extract_header(self):

        print("Extracting FITS headers...\n")

        keywords = ['ARCFILE', 'ORIGFILE', 'DATE-OBS', 'RA', 'DEC', 'OBJECT', 'MJD-OBS', \
                    'ESO OBS TARG NAME', 'ESO OBS PROG ID', 'ESO OBS ID', 'ESO OBS WATERVAPOUR', \
                    'ESO TPL ID', 'ESO DPR CATG', 'ESO DPR TECH', 'ESO DPR TYPE', 'ESO DET EXP ID', \
                    'ESO DET SEQ1 DIT', 'ESO DET NDIT', 'ESO SEQ NEXPO', 'ESO SEQ NODPOS', 'ESO SEQ NODTHROW', \
                    'ESO SEQ CUMOFFSETX', 'ESO SEQ CUMOFFSETY', 'ESO SEQ JITTERVAL', 'ESO TEL AIRM START', \
                    'ESO TEL IA FWHM', 'ESO TEL AMBI TAU0', 'ESO TEL AMBI IWV START', 'ESO INS WLEN CWLEN', \
                    'ESO INS GRAT1 ORDER', 'ESO INS WLEN ID', 'ESO INS SLIT1 NAME', 'ESO INS SLIT1 WID', \
                    'ESO INS1 OPTI1 NAME', 'ESO INS1 DROT POSANG', 'ESO INS1 FSEL ALPHA', 'ESO INS1 FSEL DELTA', \
                    'ESO AOS RTC LOOP STATE']
        keywords = ['ARCFILE', 'ESO DPR TYPE', 'ESO DET SEQ1 DIT', 'ESO INS WLEN ID']

        
        raw_files = pathlib.Path(self.rawpath).glob("*.fits")

        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:
            header = fits.getheader(file_item)

            for key_item in keywords:
                if 'FILE' in key_item:
                    if key_item in header:
                        header_dict[key_item].append(self.rawpath/header[key_item])
                    else:
                        header_dict[key_item].append(None)
                else:
                    if key_item in header:
                        header_dict[key_item].append(header[key_item])
                    else:
                        header_dict[key_item].append(None)

        for key_item in keywords:
            # column_name = key_item.replace(" ", ".")
            # column_name = column_name.replace("ESO.", "")
            self.header_info[key_item] = header_dict[key_item]
        
        self.header_info.to_csv(self.calpath/'header_info.txt', index=False, sep=' ')

    def add_to_calib(self, file_name, file_type):

        hdr = fits.getheader(file_name)
        item = [file_type, file_name]
        if hdr is not None:
            for key in self.header_keys.keys():
                if 'file' not in key:
                    item.append(hdr[self.header_keys[key]])
            self.calib_info.loc[len(self.calib_info.index)] = item
        else:
            raise RuntimeError("Header of the {} file not found.".format(file_type))
        
        self.calib_info.to_csv(self.calpath/'calib_info.txt', index=False, sep=' ')


    def cal_dark(self, clip=4, verbose=True):
        indices = self.header_info[self.key_data_type] == "DARK"

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
            for file in self.header_info[indices][indices_dit]["ARCFILE"]:
                with fits.open(file) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))
            master = np.nanmedian(dt, axis=0)
            badpix = np.zeros_like(master).astype(bool)
            for i, det in enumerate(master):
                filtered_data = sigma_clip(det, sigma=clip, axis=0)
                badpix[i] = filtered_data.mask
                master[i][badpix[i]] = np.nan
            
            file_name = self.calpath/'DARK_MASTER_DIT{}.fits'.format(item)
            su.wfits(file_name, master, hdr)
            self.add_to_calib(file_name, "DARK_MASTER")
            
            if verbose:
                print("{0:.1f} percent of pixels identified as bad.".format(np.sum(badpix)/badpix.size*100.))
            file_name = self.calpath/'DARK_BPM_DIT{}.fits'.format(item)
            su.wfits(file_name, badpix.astype(int), hdr)
            self.add_to_calib(file_name, "DARK_BPM")

    

    
    
    def cal_flat(self, verbose=True):
        indices = self.header_info[self.key_data_type] == "FLAT"
        # indices_flat = self.header_info[indices_flat][self.key_data_type] == "FLAT"

        # Check unique DIT
        unique_dit = set()
        for item in self.header_info[indices][self.key_DIT]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # master flat
        for item in unique_dit:
            indices_dit = self.header_info[indices][self.key_DIT] == item
            indices_dark = (self.calib_info['CAL.TYPE'] == "DARK_MASTER") & (self.calib_info[self.key_DIT] == item)
            indices_bpm = (self.calib_info['CAL.TYPE'] == "DARK_BPM") & (self.calib_info[self.key_DIT] == item)
            dt = []
            for file in self.header_info[indices][indices_dit]["ARCFILE"]:
                with fits.open(file) as hdu:
                    hdr = hdu[0].header
                    dt.append(np.array([hdu[i].data for i in range(1, len(hdu))]))
            master = np.nanmedian(dt, axis=0)
            assert (indices_dark.sum())<2
            for file in self.calib_info[indices_dark]["ARCFILE"]:
                dark = fits.getdata(file)
            for file in self.calib_info[indices_bpm]["ARCFILE"]:
                badpix = fits.getdata(file)
            master -= dark
            
            # order tracing
            for i, (det, bad) in enumerate(zip(master, badpix)):
                su.order_trace(det, bad)





            file_name = self.calpath/'FLAT_MASTER_DIT{}.fits'.format(item)
            su.wfits(file_name, master, hdr)
            self.add_to_calib(file_name, "FLAT_MASTER")
            
    

    def cal_wave(self, verbose=False):
        indices = self.header_data["DPR.TYPE"] == "WAVE,UNE"
        indices2 = self.header_data["DPR.TYPE"] == "WAVE,FPET"

        # Create SOF file
        sof_file = pathlib.Path(self.calpath / "wave.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ARCFILE"]:
                sof_open.write(f"{self.rawpath}/{item} WAVE_UNE\n")
            for item in self.header_data[indices2]["ARCFILE"]:
                sof_open.write(f"{self.rawpath}/{item} WAVE_FPET\n")

            file_found = False
            if "CAL_DARK_MASTER" in self.calibDB:
                for key, value in self.calibDB["CAL_DARK_MASTER"].items():
                    if not file_found and value["DIT"] == self.header_data[indices].iloc[0]["DET.SEQ1.DIT"]:
                        sof_open.write(f"{key} CAL_DARK_MASTER\n")
                        file_found = True
            if not file_found:
                raise RuntimeError("Dark master frame with DIT={} not found.".format(self.header_data[indices].iloc[0]["DET.SEQ1.DIT"]))

            file_found = False
            if "CAL_DARK_BPM" in self.calibDB:
                for key, value in self.calibDB["CAL_DARK_BPM"].items():
                    if not file_found and value["DIT"] == self.header_data[indices].iloc[0]["DET.SEQ1.DIT"]:
                        sof_open.write(f"{key} CAL_DARK_BPM\n")
                        file_found = True
            
            file_found = False
            if "UTIL_WAVE_TW" in self.calibDB:
                for key, value in self.calibDB["UTIL_WAVE_TW"].items():
                    if not file_found and value["WLEN"] == self.header_data[indices].iloc[0]["INS.WLEN.ID"]:
                        sof_open.write(f"{key} UTIL_WAVE_TW\n")
                        file_found = True
            if not file_found:
                raise RuntimeError("TW TABLE file not found.")

            file_found = False
            if "EMISSION_LINES" in self.calibDB:
                for key, value in self.calibDB["EMISSION_LINES"].items():
                    if not file_found and value["WLEN"] == self.header_data[indices].iloc[0]["INS.WLEN.ID"]:
                        sof_open.write(f"{key} EMISSION_LINES\n")
                        file_found = True
            if not file_found:
                raise RuntimeError("TW TABLE file not found.")

            sof_open.write(f"{self.detlin_path} CAL_DETLIN_COEFFS\n")

        # Run EsoRex
        esorex = [
            "esorex",
            # f"--recipe-config={config_file}",
            f"--output-dir={self.calpath}",
            "cr2res_cal_wave",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calpath, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with master flat

        fits_files = pathlib.Path(self.calpath).glob("cr2res_cal_wave_tw_*.fits")

        for item in fits_files:
            self.add_to_calib_DB("CAL_WAVE_TW", str(item))


        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.calibDB, json_file, indent=4)


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

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_combinedB.fits")
            for item in fits_files:
                self.add_to_calib_DB("STD_NODDING_COMBINEDB", str(item))

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_extractedA.fits")
            for item in fits_files:
                self.add_to_calib_DB("STD_NODDING_extractedA", str(item))

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_extractedB.fits")
            for item in fits_files:
                self.add_to_calib_DB("STD_NODDING_extractedB", str(item))
        else:
            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_combinedA.fits")
            for item in fits_files:
                self.add_to_calib_DB("OBS_NODDING_COMBINEDA", str(item))

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_combinedB.fits")
            for item in fits_files:
                self.add_to_calib_DB("OBS_NODDING_COMBINEDB", str(item))

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_extractedA.fits")
            for item in fits_files:
                self.add_to_calib_DB("OBS_NODDING_extractedA", str(item))

            fits_files = pathlib.Path(outpath).glob("cr2res_obs_nodding_extractedB.fits")
            for item in fits_files:
                self.add_to_calib_DB("OBS_NODDING_extractedB", str(item))

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

