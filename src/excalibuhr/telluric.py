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


def make_telluric_grid(savepath):

    # Set the start and end wavelength, in nm
    wavestart = 1850.0
    waveend = 2560.0

    # Set parameter ranges of the grid
    h2o_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) 
    other_range = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    teff_range = np.arange(-0.0012, 0.00121, 3e-4)

    # Make the model
    modeler = Modeler()

    all_species = ['CO', 'N2O', 'CH4', 'CO2', 'H2O']
    # Read the atmopshere profile from Molecfit
    src_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(src_path, '../../data/ATM_PROFILE_molecfit.fits')
    atm_profile = fits.getdata(filename)
    height = atm_profile['HGT']
    press = atm_profile['PRE']
    temp = atm_profile['TEM']
    modeler.EditProfile("temperature", height, temp)
    modeler.EditProfile("pressure", height, press)
    for species in all_species:
        profile = atm_profile[species]
        modeler.EditProfile(species, height, profile)

    # Forward models with TelFit
    model = modeler.MakeModel(
                angle=0.,
                humidity=0.,
                o3=0.039, 
                ch4=0.,
                co2=0., 
                n2o=0., 
                co=0., 
                o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                lat=tel_lat,
                alt=tel_alt,
                vac2air=False,
                lowfreq=1e7/waveend,
                highfreq=1e7/wavestart)
    grid_o3 = model.y
    wave = model.x

    # plt.plot(wave, model.y)
    # plt.show()

    model = modeler.MakeModel(
                angle=0.,
                humidity=0.,
                o3=0., 
                ch4=0.,
                co2=0., 
                n2o=0.32, 
                co=0., 
                o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                lat=tel_lat,
                alt=tel_alt,
                wavegrid=wave,
                vac2air=False,
                lowfreq=1e7/waveend,
                highfreq=1e7/wavestart)
    grid_n2o = model.y

    model = modeler.MakeModel(
                angle=0.,
                humidity=0.,
                o3=0., 
                ch4=0.,
                co2=0., 
                n2o=0., 
                co=0.14, 
                o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                lat=tel_lat,
                alt=tel_alt,
                wavegrid=wave,
                vac2air=False,
                lowfreq=1e7/waveend,
                highfreq=1e7/wavestart)
    grid_co = model.y


    grid_t_ch4, grid_t_h2o, grid_t_co2 = [], [], []
    for teff in teff_range:
        modeler.EditProfile("temperature", height, temp*temp_quad_perturb(height, teff))
        grid_ch4 = []
        for ch4 in other_range:
            model = modeler.MakeModel(
                        angle=0.,
                        humidity=0.,
                        o3=0., 
                        ch4=1.8*ch4,
                        co2=0., 
                        n2o=0., 
                        co=0., 
                        o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                        lat=tel_lat,
                        alt=tel_alt,
                        wavegrid=wave,
                        vac2air=False,
                        lowfreq=1e7/waveend,
                        highfreq=1e7/wavestart)
            grid_ch4.append(model.y)
        grid_t_ch4.append(grid_ch4)

        grid_co2 = []
        for co2 in other_range:
            model = modeler.MakeModel(
                        angle=0.,
                        humidity=0.,
                        o3=0., 
                        ch4=0.,
                        co2=368.5*co2, 
                        n2o=0., 
                        co=0., 
                        o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                        lat=tel_lat,
                        alt=tel_alt,
                        wavegrid=wave,
                        vac2air=False,
                        lowfreq=1e7/waveend,
                        highfreq=1e7/wavestart)
            grid_co2.append(model.y)
        grid_t_co2.append(grid_co2)

        grid_h2o = []
        for hum in h2o_range:
            model = modeler.MakeModel(
                        angle=0.,
                        humidity=hum*100.,
                        o3=0., 
                        ch4=0.,
                        co2=0., 
                        n2o=0., 
                        co=0., 
                        o2=0., no=0., so2=0., no2=0., nh3=0., hno3=0.,
                        lat=tel_lat,
                        alt=tel_alt,
                        wavegrid=wave,
                        vac2air=False,
                        lowfreq=1e7/waveend,
                        highfreq=1e7/wavestart)
            grid_h2o.append(model.y)
        grid_t_h2o.append(grid_h2o)

        
    filename = os.path.join(savepath, 'telfit_grid.fits')
    su.wfits(filename, 
            ext_list={"H2O": grid_t_h2o,
                    "CH4": grid_t_ch4,
                    "CO2": grid_t_co2,
                    "N2O": grid_n2o,
                    "CO": grid_co,
                    "O3": grid_o3,
                    "WAVE": wave
            })
