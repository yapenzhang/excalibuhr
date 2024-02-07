import os
import sys
import glob
import numpy as np 
from astropy.io import fits
# from telfit import Modeler
import excalibuhr.utils as su
from scipy.interpolate import RegularGridInterpolator

class TelluricGrid:
    """
    Telluric transmission spectra grid generated with TelFit
    https://telfit.readthedocs.io/en/latest/
    """

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

        try:
            self.load_grid()
        except:
            self.make_grid()
            self.combine_grid()
            self.load_grid()


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


    def make_grid(self):
        print("Generate telluric grid spectra...")

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
        print(f"Load telluric grid spectra from {filename}...")
        hdul = fits.open(filename)
        grid = {}
        for i, hdu in enumerate(hdul):
            if i > 0:
                grid[hdu.name] = hdu.data
        self.grid = grid
    


class SonoraGrid:
    """
    Sonora brown dwarf and substellar model grid downloaded from
    https://zenodo.org/record/5063476#.Y7QLg3aZPEY
    
    """

    def __init__(self, gridpath=None):

        if gridpath is None:
            gridpath = os.path.dirname(os.path.abspath(__file__))
            gridpath = os.path.join(gridpath, '../', '../', 'data')

        self.gridpath = gridpath

        # Set parameter ranges of the grid
        self.teff_grid = np.arange(200, 600, 25) 
        self.teff_grid = np.append(self.teff_grid, np.arange(600, 1000, 50))
        self.teff_grid = np.append(self.teff_grid, np.arange(1000, 2500, 100))

        self.gravity_grid = np.array([10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])
        self.logg_grid = np.log10(np.array([10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160])*1e2)

        # self.metal_grid = np.array([0, 0.25, 0.5, 0.75, 1.0])

        try:
            self.load_grid()
        except:
            self.make_grid()
            self.load_grid()


    def make_grid(self):
        print("Generate grid profiles...")
        nlayer = 91
        ncolumn = 2 #t, p
        grid = np.zeros((len(self.teff_grid), 
                               len(self.gravity_grid), 
                            #    len(self.metal_grid), 
                               ncolumn, nlayer))
        for i, teff in enumerate(self.teff_grid):
            for j, grav in enumerate(self.gravity_grid):
                # for k, z in enumerate(self.metal_grid):
                    if grav == 10:
                        filename = f't{teff:.0f}g{grav:.0f}nc_m+0.0.dat'
                    else:
                        filename = f't{teff:.0f}g{grav:.0f}nc_m0.0.dat'
                    filename = os.path.join(self.gridpath, filename)
                    if not os.path.exists(filename):
                        raise Exception(f"Model file {filename} not found.")
                    pt = np.genfromtxt(filename, skip_header=1)
                    t = pt[:,2]
                    p = pt[:,1]
                    grid[i,j,:] = [t, p]

        np.save(os.path.join(self.gridpath,'sonora_PT_grid'), grid)

    def load_grid(self):
        filename = os.path.join(self.gridpath, 'sonora_PT_grid.npy')
        file = glob.glob(filename)
        print(f"Load sonora grid profiles from {filename}...")
        self.grid = np.load(file[0])
        
    def interp_grid(self):
        interp = RegularGridInterpolator(
                    (self.teff_grid, self.gravity_grid), 
                    self.grid, bounds_error=False, fill_value=None)
        return interp
    
    def interp_PT(self, teff, logg):
        pt = self.interp_grid()([teff, 1e1**logg*1e-2])
        t = pt[0,0]
        p = pt[0,1]
        return p, t


class StellarGrid:
    """
    MARCS stellar LTE model grid downloaded from
    https://marcs.oreme.org/
    """

    def __init__(self, gridpath=None):

        if gridpath is None:
            gridpath = os.path.dirname(os.path.abspath(__file__))
            gridpath = os.path.join(gridpath, '../', '../', 'data')

        self.gridpath = gridpath

        # Set parameter ranges of the grid
        self.teff_grid = np.arange(2500, 4000, 100)
        self.teff_grid = np.append(self.teff_grid, np.arange(4000, 8000, 250))

        self.logg_grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        # self.metal_grid = np.array([0, 0.25, 0.5, 0.75, 1.0])

        try:
            self.load_grid()
        except:
            self.make_grid()
            self.load_grid()


    def make_grid(self):
        print("Generate stellar grid profiles...")
        nlayer = 56
        ncolumn = 4 #t, p, h2o, co
        marcs_grid = np.zeros((len(self.teff_grid), 
                               len(self.logg_grid), 
                            #    len(self.metal_grid), 
                               ncolumn, nlayer))
        for i, teff in enumerate(self.teff_grid):
            for j, logg in enumerate(self.logg_grid):
                # for k, z in enumerate(self.metal_grid):
                    z = 0.
                    filename = f'p{teff:.0f}_g+{logg:.1f}_m0.0_t00_st_z+{z:.2f}_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod'
                    filename = os.path.join(self.gridpath, filename)
                    if not os.path.exists(filename):
                        raise Exception(f"Marcs model file '{filename}' not found.")
                    pt = np.genfromtxt(filename, skip_header=25, skip_footer=229)
                    molec = np.genfromtxt(filename, skip_header=140, skip_footer=114)
                    t = pt[:,4]
                    p = pt[:,6]*1e-6
                    # vmr_H2 = 1e1**molec[:,4]*1e-6/p
                    vmr_H2O = 1e1**molec[:,6]*1e-6/p
                    vmr_CO = 1e1**molec[:,9]*1e-6/p
                    marcs_grid[i,j,:] = [t, p, vmr_H2O, vmr_CO]

        np.save(os.path.join(self.gridpath,'marcs_grid'), marcs_grid)

    def load_grid(self):
        filename = os.path.join(self.gridpath, '*_grid.npy')
        file = glob.glob(filename)
        print(f"Load stellar grid profiles from {filename}...")
        self.grid = np.load(file[0])
        
    def interp_grid(self):
        interp = RegularGridInterpolator(
                    (self.teff_grid, self.logg_grid), 
                    self.grid, bounds_error=False, fill_value=None)
        return interp
    
    def interp_PT(self, teff, logg):
        pt = self.interp_grid()([teff, logg])
        t = pt[0,0]
        p = pt[0,1]
        return p, t
    

class LimbDarkGrid:

    """
    Limb darkening coefficients downloaded from
    https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/546/A14
    """
    def __init__(self, gridpath=None):

        if gridpath is None:
            gridpath = os.path.dirname(os.path.abspath(__file__))
            gridpath = os.path.join(gridpath, '../', '../', 'data')
        
        self.gridpath = gridpath

        # Set parameter ranges of the grid
        self.teff_grid = np.arange(1500, 4700, 100)
        self.logg_grid = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])

        try:
            self.load_grid()
        except:
            self.make_grid()
            self.load_grid()


    def make_grid(self):
        try:
            dt = np.genfromtxt(os.path.join(self.gridpath, 'limb_dark_coeff.dat'), 
                           skip_header=10761) # Ks band
        except:
            raise Exception("Limb darkening coeff file not found.")
        us = dt[:,4].reshape(len(self.teff_grid), len(self.logg_grid))
        np.save(os.path.join(self.gridpath,'limb_dark_coeff_u'), us)

    def load_grid(self):
        filename = os.path.join(self.gridpath, '*_coeff_u.npy')
        file = glob.glob(filename)
        print(f"Load stellar limb darkening grid from {file}...")
        self.grid = np.load(file[0])
        
    def interp_grid(self):
        interp = RegularGridInterpolator((self.teff_grid, self.logg_grid), 
                            self.grid, bounds_error=False, fill_value=None)
        return interp
    
    def interp_limb(self, teff, logg):
        teff = max(teff, self.teff_grid[-1])
        teff = min(teff, self.teff_grid[0])
        logg = max(logg, self.logg_grid[-1])
        logg = min(logg, self.logg_grid[0])
        return self.interp_grid()([teff, logg])[0]