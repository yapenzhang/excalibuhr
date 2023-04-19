import os
import sys
import numpy as np 
from astropy.io import fits
from telfit import Modeler
import pylab as plt
import excalibuhr.utils as su

# telescope altitude and latitude
tel_alt = 2.635 #km
tel_lat = -24.6

def temp_quad_perturb(x, a):
    """
    Quadratic function to be applied to the nominal temperature profile,
    while keeping the temperature fixed at the height of H0 and H1.
    Here we only modify the temperature in the troposphere regime.
    """
    H0 = tel_alt #telescope altitude
    H1 = 20 #20km
    quad = a*(x**2-(H0+H1)*x+H0*H1)+1
    quad[(x<H0)|(x>H1)] = 1. 
    return quad


def make_telluric_grid(savepath, 
                       wave_range=[1850,2560],
                       free_species=['H2O','CH4','CO2'],
                       ):

    # Set the start and end wavelength, in nm
    wavestart = wave_range[0] 
    waveend =  wave_range[1] 

    nominal_ppmv = {
        'H2O': 100.,
        'CO2': 368.5, 
        'O3': 3.9e-2, 
        'N2O': 0.32, 
        'CO': 0.14,
        'CH4': 1.8,
        'O2': 2.1e5,
    }
    all_species = nominal_ppmv.keys()
    fixed_species = [s for s in all_species if s not in free_species]

    # Set parameter ranges of the grid
    h2o_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) 
    other_range = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    teff_range = np.arange(-0.0012, 0.00121, 3e-4)

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

    grid = {}
    # Forward models with TelFit
    model = modeler.MakeModel(
                angle=0.,
                humidity=0.,
                o3=nominal_ppmv['O3'], 
                ch4=0.,
                co2=0., 
                n2o=0., 
                co=0., 
                o2=0., 
                no=0., so2=0., no2=0., nh3=0., hno3=0.,
                lat=tel_lat,
                alt=tel_alt,
                vac2air=False,
                lowfreq=1e7/waveend,
                highfreq=1e7/wavestart)
    wave = model.x 
    grid['O3'] = model.y
    grid['WAVE'] = model.x

    for species in fixed_species:
        if species != 'O3':
            values = [nominal_ppmv[s]*float(s == species) for s in all_species]
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
                    lat=tel_lat,
                    alt=tel_alt,
                    wavegrid=wave,
                    vac2air=False,
                    lowfreq=1e7/waveend,
                    highfreq=1e7/wavestart)
            grid[species] = model.y
    
    for species in free_species:
        grid_sp_t = []
        for teff in teff_range:
            grid_sp = []
            modeler.EditProfile("temperature", height, temp*temp_quad_perturb(height, teff))
            if species == 'H2O':
                ppmv_range = h2o_range
            else:
                ppmv_range = other_range
            for ppmv in ppmv_range:
                values = [nominal_ppmv[s]*float(s == species)*ppmv for s in all_species]
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
                        lat=tel_lat,
                        alt=tel_alt,
                        vac2air=False,
                        lowfreq=1e7/waveend,
                        highfreq=1e7/wavestart)
                grid_sp.append(model.y)
            grid_sp_t.append(grid_sp)
        grid[species] = grid_sp_t

    filename = os.path.join(savepath, 'telfit_grid.fits')
    su.wfits(filename, ext_list=grid)
