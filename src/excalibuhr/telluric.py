import os
import sys
import numpy as np 
from astropy.io import fits
# from telfit import Modeler
import excalibuhr.utils as su

class TelluricGrid:

    def __init__(self,
                savepath, 
                wave_range=[1850,2560],
                free_species=['CH4','CO2','H2O'],
                 ):
        
        self.savepath = savepath
        
        # telescope altitude and latitude
        self.tel_alt = 2.635 #km
        self.tel_lat = -24.6

        # Set the start and end wavelength, in nm
        self.wavestart = wave_range[0] 
        self.waveend =  wave_range[1] 

        self.nominal_ppmv = {
            'H2O': 100.,
            'CO2': 368.5, 
            'O3': 3.9e-2, 
            'N2O': 0.32, 
            'CO': 0.14,
            'CH4': 1.8,
            'O2': 2.1e5,
        }
        self.all_species = self.nominal_ppmv.keys()
        self.free_species = free_species

        # Set parameter ranges of the grid
        self.humidity_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) 
        self.ppmv_range = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
        self.temp_range = np.arange(-0.0012, 0.00121, 4e-4)


    def temp_quad_perturb(self, x, a):
        """
        Quadratic function to be applied to the nominal temperature profile,
        while keeping the temperature fixed at the height of H0 and H1.
        Here we only modify the temperature in the troposphere regime.
        """
        H0 = self.tel_alt #telescope altitude
        H1 = 20 #20km
        quad = a*(x**2-(H0+H1)*x+H0*H1)+1
        quad[(x<H0)|(x>H1)] = 1. 
        return quad


    def make_telluric_grid(self):

        # Make the model
        modeler = Modeler()

        # Read the atmopshere profile from Molecfit
        src_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(src_path, '../../data/ATM_PROFILE_molecfit.fits')
        atm_profile = fits.getdata(filename)
        height = atm_profile['HGT']
        press = atm_profile['PRE']
        temp = atm_profile['TEM']
        modeler.EditProfile("temperature", height, temp)
        modeler.EditProfile("pressure", height, press)
        profile_change_species = ['CH4', 'CO2', 'H2O', 'O2', 'CO']
        for species in profile_change_species:
            profile = atm_profile[species]
            modeler.EditProfile(species, height, profile)

        # Forward models with TelFit
        model = modeler.MakeModel(
                    angle=0.,
                    humidity=0.,
                    o3=self.nominal_ppmv['O3'], 
                    ch4=0.,
                    co2=0., 
                    n2o=0., 
                    co=0., 
                    o2=0., 
                    no=0., so2=0., no2=0., nh3=0., hno3=0.,
                    lat=self.tel_lat,
                    alt=self.tel_alt,
                    vac2air=False,
                    lowfreq=1e7/self.waveend,
                    highfreq=1e7/self.wavestart)
        wave = model.x 
        filename = os.path.join(self.savepath, 'telfit_WAVE.fits')
        su.wfits(filename, ext_list={'WAVE': model.x})
        filename = os.path.join(self.savepath, 'telfit_O3.fits')
        su.wfits(filename, ext_list={'FLUX': model.y})

        fixed_species = [s for s in self.all_species if s not in self.free_species + ['O3']]
        for species in fixed_species:
            values = [self.nominal_ppmv[s]*float(s == species) for s in self.all_species]
            model = modeler.MakeModel(
                    angle=0.,
                    humidity=values[0],
                    co2=values[1], 
                    o3=values[2], 
                    n2o=values[3], 
                    co=values[4], 
                    ch4=values[5],
                    o2=values[6], 
                    no=0., so2=0., no2=0., nh3=0., hno3=0.,
                    lat=self.tel_lat,
                    alt=self.tel_alt,
                    wavegrid=wave,
                    vac2air=False,
                    lowfreq=1e7/self.waveend,
                    highfreq=1e7/self.wavestart)
            # grid[species] = model.y
            filename = os.path.join(self.savepath, f'telfit_{species}.fits')
            su.wfits(filename, ext_list={'FLUX': model.y})
        
        for species in self.free_species:
            grid_sp_t = []
            for teff in self.temp_range:
                grid_sp = []
                modeler.EditProfile("temperature", height, temp*self.temp_quad_perturb(height, teff))
                if species == 'H2O':
                    ppmv_range = self.humidity_range
                else:
                    ppmv_range = self.ppmv_range
                for ppmv in ppmv_range:
                    values = [self.nominal_ppmv[s]*float(s == species)*ppmv for s in self.all_species]
                    print(species, teff, ppmv)
                    model = modeler.MakeModel(
                            angle=0.,
                            humidity=values[0],
                            co2=values[1], 
                            o3=values[2], 
                            n2o=values[3], 
                            co=values[4], 
                            ch4=values[5],
                            o2=values[6], 
                            no=0., so2=0., no2=0., nh3=0., hno3=0.,
                            lat=self.tel_lat,
                            alt=self.tel_alt,
                            wavegrid=wave,
                            vac2air=False,
                            lowfreq=1e7/self.waveend,
                            highfreq=1e7/self.wavestart)
                    grid_sp.append(model.y)
                grid_sp_t.append(grid_sp)
            filename = os.path.join(self.savepath, f'telfit_{species}.fits')
            su.wfits(filename, ext_list={'FLUX': grid_sp_t})


    def combine_grid(self):
        grid = {}
        file_tmp = []
        for species in list(self.all_species) + ['WAVE']:
            filename = os.path.join(self.savepath, f'telfit_{species}.fits')
            dt = fits.getdata(filename)
            grid[species] = dt 
            file_tmp.append(filename)
        filename = os.path.join(self.savepath, 'telfit_grid.fits')
        su.wfits(filename, ext_list=grid)
        os.remove(file_tmp)

    def load_grid(self):
        filename = os.path.join(self.savepath, 'telfit_grid.fits')
        hdul = fits.open(filename)
        grid = {}
        for i, hdu in enumerate(hdul):
            if i > 0:
                grid[hdu.name] = hdu.data
        return grid