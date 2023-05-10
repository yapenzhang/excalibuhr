import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import copy
import json
import pickle
import warnings
import numpy as np
import urllib.request
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator#, CubicSpline
from scipy.ndimage import gaussian_filter
# from numpy.polynomial import polynomial as Poly
from numpy.polynomial import chebyshev as Chev
from astropy.io import fits
from PyAstronomy import pyasl
import pymultinest
import corner
import excalibuhr.utils as su 
from excalibuhr.data import SPEC2D
from excalibuhr.grids import TelluricGrid, StellarGrid, LimbDarkGrid
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw
from petitRADTRANS.retrieval.util import getMM, calc_MMW
import petitRADTRANS.poor_mans_nonequ_chem as pm
# from petitRADTRANS.retrieval import cloud_cond as fc
# from typeguard import typechecked
from sksparse.cholmod import cholesky
import matplotlib.pyplot as plt 
cmap = plt.get_cmap("tab10")
cmap_hue = plt.get_cmap("tab20c")

# def getMM(species):
#     """
#     Get the molecular mass of a given species.

#     This function uses the molmass package to
#     calculate the mass number for the standard
#     isotope of an input species. If all_iso
#     is part of the input, it will return the
#     mean molar mass.

#     Args:
#         species : string
#             The chemical formula of the compound. ie C2H2 or H2O
#     Returns:
#         The molar mass of the compound in atomic mass units.
#     """
#     from molmass import Formula

#     e_molar_mass = 5.4857990888e-4  # (g.mol-1) e- molar mass (source: NIST CODATA)

#     if species == 'e-':
#         return e_molar_mass
#     elif species == 'H-':
#         return Formula('H').mass + e_molar_mass

#     name = species.split("_")[0]
#     name = name.split(',')[0]
#     f = Formula(name)

#     if "all_iso" in species:
#         return f.mass
    
#     return f.isotope.massnumber

def calc_abundance_ratio_posterior(a, b, params, samples):
    tol = 1e-20
    MMW = 2.33
    elemental_solar = {
        'H': 12,
        'He': 10.914,
        'C': 8.46,
        'N': 7.83,
        'O': 8.69,
        'Na': 6.22,
        'Ca': 6.30,
        'Fe': 7.46,
    }
    from molmass import Formula
    abund_a, abund_b = [], []
    for i, key in enumerate(params):
        if key.split("_")[0] == 'logX':
            species = ''.join(key.split("_")[1:])
            if '36' in species:
                species = '[13C]O'
            f = Formula(species)
            df = f.composition().dataframe()
            if a in df.index:
                abund_a.append(df.loc[a, 'Count']*1e1**samples[:,i]/getMM(species)*MMW)
            if b in df.index:
                if species == 'H2O':
                    abund_b.append(df.loc[b, 'Count']*1e1**samples[:,i]/getMM(species)*MMW*1.)
                else:
                    abund_b.append(df.loc[b, 'Count']*1e1**samples[:,i]/getMM(species)*MMW)
    abund_a = np.sum(abund_a, axis=0)
    abund_b = np.sum(abund_b, axis=0)
    if b == 'H':
        abund_b += 2. * 0.84
        ratio = np.log10(abund_a/abund_b) - (elemental_solar[a] - 12)
    else:
        ratio = abund_a/(abund_b+tol)
    return ratio



def calc_elemental_ratio(a, b, abundances):
    tol = 1e-20
    from molmass import Formula
    abund_a, abund_b = [], []
    for key in abundances:
        species = key.split("_")[0]
        f = Formula(species)
        df = f.composition().dataframe()
        if a in df.index:
            abund_a.append(df.loc[a, 'Count']*abundances[key])
        if b in df.index:
            abund_b.append(df.loc[b, 'Count']*abundances[key])
    abund_a = np.sum(abund_a, axis=0)
    abund_b = np.sum(abund_b, axis=0)
    return abund_a/(abund_b+tol)


def calc_mass_to_mol_frac(abundances, MMW):
    mol_fraction = {}
    for key in abundances:
        if "36" in key.split("_"):
            mass = getMM('[13C]O')
        else:
            mass = getMM(key)
        mol_fraction[key] = abundances[key] / mass * MMW
    return mol_fraction

# def calc_MMW(abundances):
#     """
#     calc_MMW
#     Calculate the mean molecular weight in each layer.

#     Args:
#         abundances : dict
#             dictionary of abundance arrays, each array must have the shape of the pressure array used in pRT,
#             and contain the abundance at each layer in the atmosphere.
#     """
#     mmw = sys.float_info.min  # prevent division by 0

#     for key in abundances.keys():
#         # exo_k resolution
#         spec = key.split("_")[0]
#         mmw += abundances[key] / getMM(spec)

#     return 1.0 / mmw

class Parameter:

    def __init__(self, name, value=None, prior=(0,1), is_free=True, prior_type='U'):
        self.name = name
        self.is_free = is_free
        self.value = value
        self.prior = prior
        self.prior_type = prior_type

    def set_value(self, value):
        self.value = value

    def set_prior(self, prior):
        self.prior = prior


class Prior:
    """
    Taken from MultiNEST priors.f90
    Usage e.g.:
    from priors import Priors
    pri=Priors()
    cube[0]=pri.UniformPrior(cube[0],1.0,199.0)
    cube[0]=pri.GeneralPrior(cube[0],'LOG',1.0,100.0)
    etc.
    """

    def __init__(self):
        pass

    def GeneralPrior(self,r,PriorType,x1,x2):
        if PriorType=='DELTA':
            return self.DeltaFunctionPrior(r,x1,x2)
        elif PriorType=='U':
            return self.UniformPrior(r,x1,x2)
        elif PriorType=='LOG':
            return self.LogPrior(r,x1,x2)
        elif PriorType=='GAUSS':
            return self.GaussianPrior(r,x1,x2)
        elif PriorType=='GAMMA':
            return self.InvGammaPrior(r,x1,x2)
        else:
            raise Exception('Unrecognised prior')

    def DeltaFunctionPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Delta[x1]"""
        return x1

    def UniformPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Uniform[x1:x2]"""
        return x1+r*(x2-x1)

    def LogPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  LogUniform[x1:x2]"""
        from math import log10
        if (r <= 0.0):
                return -1.0e32
        else:
            lx1=log10(x1); lx2=log10(x2)
            return 10.0**(lx1+r*(lx2-lx1))

    def GaussianPrior(self,r,mu,sigma):
        """Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
        from math import sqrt
        from scipy.special import erfcinv
        if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
            return -1.0e32
        else:
            return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r))
    
    def InvGammaPrior(self,r,alpha,beta):
        """Uniform[0:1]  ->  InvGamma[]"""
        from scipy.stats import invgamma
        rvs = invgamma(alpha, scale=beta)
        return rvs.ppf(r)


class Photometry:

    def __init__(self, filter_name, f_lambda, f_err=None):

        self.name = 'photometry'
        self.mode = 'c-k'
        self.filter_name = filter_name
        self.f_lambda = f_lambda
        self.f_err = f_err
        self.get_filter_transm()

    def get_filter_transm(self):
        
        url = "http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id="+ self.filter_name
        urllib.request.urlretrieve(url, "filter.dat")
        transmission = np.genfromtxt('filter.dat')
        transmission[:,0] *= 1e-1 # wavelength in nm
        if transmission.size == 0:
            raise ValueError("The filter data of {} could not be downloaded".format(self.filter_name))
        os.remove('filter.dat')

        self.wlen = transmission[:,0]
        self.flux = transmission[:,1]

    def make_wlen_bins(self):
        data_wlen_bins = np.zeros_like(self.wlen)
        data_wlen_bins[:-1] = np.diff(self.wlen)
        data_wlen_bins[-1] = data_wlen_bins[-2]
        self.wlen_bins = data_wlen_bins


class ObsData(SPEC2D):

    def __init__(self, wlen=None, flux=None, err=None, filename=None, 
                 name=None, R=None):
        
        super().__init__(wlen=wlen, flux=flux, err=err, filename=filename)
        
        self.name = name
        self.R = R

        if self.R > 1000.:
            self.mode = 'lbl'
        else:
            self.mode = 'c-k'

        if self.name.lower() == 'crires':
            self.detector_bin = 3
        else:
            self.detector_bin = 1
        



class Retrieval:

    def __init__(self, retrieval_name, out_dir):

        self.retrieval_name = retrieval_name
        self.out_dir = out_dir
        self.prefix = self.out_dir+'/'+self.retrieval_name+'_'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.obs = {} 
        self.params = {}
        self.model_tellu = lambda x: np.ones_like(x)
        self.debug = False

        attr_keys = {
                'key_radius': 'radius',
                'key_rv': 'vsys',
                'key_spin': 'vsini',
                'key_limb_dark_u': 'limb',
                'key_distance': 'distance',
                'key_airmass': 'airmass',
                'key_carbon_to_oxygen': 'C/O',
                'key_carbon_iso': '13/12C',
                'key_metallicity': '[C/H]',
                'key_gravity': 'logg',
                'key_teff': 'T_eff',
                'key_tellu_temp': 'tellu_temp',
                'key_t_bottom': "t_00",
                'key_quench': 'log_P_quench',
                'key_molecule_tolerance': 'beta',
                'key_teff_tolerance': 'tol',
                'key_penalty_param': "gamma",
                'key_spot_coverage': "f_spot",
                # 'key_spot_coverage': ""
                'key_teff_spot': 'T_spot',
            }
        for par in attr_keys.keys():
            setattr(self, par, attr_keys[par])

    
    def add_observation(self, obs_list):
        for obs in obs_list:
            obs.make_wlen_bins()
            if obs.name == 'photometry':
                if obs.name not in self.obs:
                    self.obs[obs.name] = [obs]
                else:
                    self.obs[obs.name].append(obs)
            else:
                self.obs[obs.name] = obs



    def add_parameter(self, name, value=None, prior=(0,1), is_free=True):
        self.params[name] = Parameter(name, value, prior, is_free)
        print(f"Add parameter - {name}, value: {value}, prior: {prior}, is_free: {is_free}")
        


    def add_pRT_objects(self,
                       rayleigh_species=['H2', 'He'], 
                       continuum_opacities=['H2-H2', 'H2-He'],
                       cloud_species=None,
                       lbl_opacity_sampling=5,
                       ):
        if cloud_species is not None:
            do_scat_emis = True
        else:
            do_scat_emis = False

        self.pRT_object = {}
        for instrument in self.obs.keys():
            self.pRT_object[instrument] = []
            if instrument == 'photometry':
                for data_object in self.obs[instrument]:
                    dw = (data_object.wlen[1]-data_object.wlen[0])*5.
                    wlen_cut = [(data_object.wlen[0]-dw)*1e-3, (data_object.wlen[-1]+dw)*1e-3]
                    # print(wlen_cut)
                    rt_object = Radtrans(
                            line_species=self.line_species_ck,
                            rayleigh_species=rayleigh_species,
                            continuum_opacities=continuum_opacities,
                            cloud_species=cloud_species,
                            mode=data_object.mode,
                            wlen_bords_micron=wlen_cut,
                            lbl_opacity_sampling=lbl_opacity_sampling,
                            do_scat_emis=do_scat_emis,
                        )
                    self.pRT_object[instrument].append(rt_object)
            else:
                data_object = self.obs[instrument]
                for i in range(0, data_object.Nchip, data_object.detector_bin):
                    wave_tmp = data_object.wlen[i:i+data_object.detector_bin].flatten()
                    # set pRT wavelength range sparing 200 pixels 
                    # beyond the data wavelengths for each order
                    dw = (wave_tmp[1]-wave_tmp[0])*200.
                    wlen_cut = [(wave_tmp[0]-dw)*1e-3, (wave_tmp[-1]+dw)*1e-3]
                    rt_object = Radtrans(
                            line_species=self.line_species,
                            rayleigh_species=rayleigh_species,
                            continuum_opacities=continuum_opacities,
                            cloud_species=cloud_species,
                            mode=data_object.mode,
                            wlen_bords_micron=wlen_cut,
                            lbl_opacity_sampling=lbl_opacity_sampling,
                            do_scat_emis=do_scat_emis,
                        )
                    self.pRT_object[instrument].append(rt_object)
        
    

    def add_free_PT_model(self, N_t_knots):

        for i in range(N_t_knots):
            self.add_parameter(f't_{i:02}')
        
        if self.PT_penalty_order:
            self.add_parameter(self.key_penalty_param, prior=(-2,9))

        self.PT_model = self.free_PT_model


    def free_PT_model(self):
        p_ret = np.copy(self.press)
        t_names = [x for x in self.params if x.split('_')[0]=='t']
        t_names.sort(reverse=True)
        knots_t = [self.params[x].value for x in t_names]
        knots_p = np.logspace(np.log10(self.press[0]),np.log10(self.press[-1]), len(knots_t))
        # interpolation and penalty in logT - logP space
        t_spline = splrep(np.log10(knots_p), np.log10(knots_t), k=3)
        knots, coeffs, _ = t_spline
        tret = splev(np.log10(p_ret), t_spline, der=0)
        self.temp = 1e1 ** tret

        # p_ret = np.copy(self.press)
        # t_names = [x for x in self.params if x.split('_')[0]=='t']
        # t_names.sort(reverse=True)
        # knots_t = [self.params[x].value for x in t_names]
        # knots_p = np.logspace(np.log10(self.press[0]),np.log10(self.press[-1]), len(knots_t))
        # t_spline = splrep(np.log10(knots_p), knots_t, k=1)
        # knots, coeffs, _ = t_spline
        # tret = splev(np.log10(p_ret), t_spline, der=0)
        # t_smooth = gaussian_filter(tret, 1.5)
        # self.temp = t_smooth

        if self.PT_penalty_order:

            if self.key_penalty_param in self.params:
                gamma = 1e1**self.params[self.key_penalty_param].value
            else: 
                raise Exception("PT penalty parameter not set.")
        
            # Compute the log-likelihood penalty based on the wiggliness
            # (Inverted) weight matrices, scaling the penalty of small/large segments
            inv_W_1 = np.diag(1/(1/3 * np.array([knots[i+3]-knots[i] \
                                                for i in range(1, len(knots)-4)]))
                            )
            inv_W_2 = np.diag(1/(1/2 * np.array([knots[i+2]-knots[i] \
                                                for i in range(2, len(knots)-4)]))
                            )
            inv_W_3 = np.diag(1/(1/1 * np.array([knots[i+1]-knots[i] \
                                                for i in range(3, len(knots)-4)]))
                            )

            # Fundamental difference matrix
            delta = np.zeros((len(inv_W_1), len(inv_W_1)+1))
            delta[:,:-1] += np.diag([-1]*len(inv_W_1))
            delta[:,1:]  += np.diag([+1]*len(inv_W_1))

            # 1st, 2nd, 3rd order general difference matrices
            D_1 = np.dot(inv_W_1, delta)
            D_2 = np.dot(inv_W_2, np.dot(delta[1:,1:], D_1))
            D_3 = np.dot(inv_W_3, np.dot(delta[2:,2:], D_2))
            
            # General difference penalty, computed with L2-norm
            if self.PT_penalty_order == 1:
                gen_diff_penalty = np.nansum(np.dot(D_1, coeffs[:-4])**2)
            elif self.PT_penalty_order == 2:
                gen_diff_penalty = np.nansum(np.dot(D_2, coeffs[:-4])**2)
            elif self.PT_penalty_order == 3:
                gen_diff_penalty = np.nansum(np.dot(D_3, coeffs[:-4])**2)

            self.ln_L_penalty = -(1/2*gen_diff_penalty/gamma + \
                                1/2*np.log(2*np.pi*gamma)
                                )
        else:
            self.ln_L_penalty = 0.

        return self.press, self.temp


    def add_grid_PT_model(self,
                          grid_path=None,
                          inhomo=False,
                          ):
        self.PT_model = self.grid_PT_model

        self.stellar_grid = StellarGrid(grid_path)
        self.limb_dark_grid = LimbDarkGrid(grid_path)

        teff_grid = self.stellar_grid.teff_grid
        logg_grid = self.stellar_grid.logg_grid
        metal_grid = self.stellar_grid.metal_grid

        self.add_parameter(self.key_teff, prior=(teff_grid[0], teff_grid[-1]))
        self.add_parameter(self.key_gravity, prior=(logg_grid[0], logg_grid[-1]))
        self.add_parameter(self.key_metallicity, prior=(metal_grid[0], metal_grid[-1]))

        if inhomo:
            self.add_parameter(self.key_teff_spot, prior=(0.8, 1.))
            self.add_parameter(self.key_spot_coverage, prior=(0, 0.5))


    def grid_PT_model(self):

        grid_param = np.array([self.params[self.key_teff].value, 
                               self.params[self.key_gravity].value,
                               self.params[self.key_metallicity].value,])
        self.temp = self.stellar_grid.interp_grid()(grid_param)[0,0]
        self.press = self.stellar_grid.interp_grid()(grid_param)[0,1]

        # interpolation of limb darkening
        self.limb = self.limb_dark_grid.interp_grid()(grid_param[:2])[0]
        self.ln_L_penalty = 0.

        if self.key_teff_spot in self.params:
            grid_param = np.array([self.params[self.key_teff_spot].value, 
                                    self.params[self.key_gravity].value,
                                    self.params[self.key_metallicity].value,])
            temp =  self.stellar_grid.interp_grid()(grid_param)[0,0]
            press = self.stellar_grid.interp_grid()(grid_param)[0,1]

            return [self.press, press], [self.temp, temp]

        else:
            return self.press, self.temp

    def add_free_chem_model(self):

        for species in self.line_species:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            self.add_parameter(param_name, value=-20, prior=(-12, -2))
        self.chem_model = self.free_chem_model
    
    def free_chem_model(self):

        abundances = {}
        for species in self.line_species:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            abundances[species] = np.ones_like(self.press) * 1e1 ** self.params[param_name].value

        sum_masses = 0
        for species in abundances.keys():
            sum_masses += abundances[species][0]
        massH2He = 1. - sum_masses
        abundances['H2'] = 2.*0.84/(4*0.16+2*0.84) * massH2He * np.ones_like(self.press)
        abundances['He'] = 4.*0.16/(4*0.16+2*0.84) * massH2He * np.ones_like(self.press)

        MMW = calc_MMW(abundances)*np.ones_like(self.press)

        # set abundances for corr-k species
        for species in self.line_species_ck:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            abundances[species] = np.ones_like(self.press) * 1e1 ** self.params[param_name].value

        if self.debug:
            c2o = calc_elemental_ratio('C', 'O', 
                        calc_mass_to_mol_frac(abundances, MMW))
            co_iso_name = []
            for key in abundances:
                if key.split('_')[0] == 'CO':
                    co_iso_name.append(key)
            co_iso_name.sort()
            c_iso = abundances[co_iso_name[1]]/abundances[co_iso_name[0]]
            print("C/O: ", np.mean(c2o))
            print("12CO/13CO: ", np.mean(c_iso))

        return abundances, MMW


    def add_equ_chem_model(self):

        self.add_parameter(self.key_carbon_iso, prior=(-12, 0))
        self.add_parameter(self.key_carbon_to_oxygen, prior=(0.1, 1.5))
        self.add_parameter(self.key_metallicity, prior=(-1.5, 1.5))
        self.add_parameter(self.key_quench, value=-10, 
                           prior=(np.log10(self.press[0]), np.log10(self.press[-1])))
        
        self.chem_model = self.equ_chem_model

    def equ_chem_model(self):

        COs = self.params[self.key_carbon_to_oxygen].value * np.ones_like(self.press)
        FeHs = self.params[self.key_metallicity].value * np.ones_like(self.press)
        abund_interp = pm.interpol_abundances(
                    COs, FeHs, 
                    self.temp, self.press,
                    Pquench_carbon = 1e1**self.params[self.key_quench].value,
                    )
        MMW = abund_interp['MMW']

        abundances = {}
        abundances['H2'] = abund_interp['H2']
        abundances['He'] = abund_interp['He']
        for species in self.line_species + self.line_species_ck:
            if '36' in species.split('_'):
                abundances[species] = abund_interp[species.split('_')[0]] \
                                       * 1e1**self.params[self.key_carbon_iso].value
            else:
                abundances[species] = abund_interp[species.split('_')[0]]

        return abundances, MMW
        
    
    def add_cloud_model(self):
        pass
    
        
    
    def forward_model_pRT(self, leave_out=None, contribution=False):

        # get temperarure profile
        press, temp = self.PT_model()

        # if self.debug:
        #     plt.plot(self.temp, self.press)
        #     plt.plot(temp[1], press[1])
        #     plt.ylim([self.press[-1],self.press[0]])
        #     plt.yscale('log')
        #     plt.xlabel('T (K)')
        #     plt.ylabel('Pressure (bar)')
        #     plt.show()
        #     plt.clf()

        # get chemical abundances and mean molecular weight
        self.abundances, self.MMW = self.chem_model()

        # leave out one species
        if leave_out is not None:
            for key in self.abundances:
                if leave_out in key:
                    self.abundances[key] = 1e-30 * np.ones_like(self.press)

        self.model_native = {}
        for instrument in self.obs.keys():
            model = []
            for rt_object in self.pRT_object[instrument]:
                # if calculate inhomogeneous model due to stellar spots,
                # linear combine models according to the coverage parameter
                if self.key_spot_coverage in self.params:
                    f_spot = self.params[self.key_spot_coverage].value
                    f_lambda = []
                    for pressure, temperature in zip(press, temp):
                        rt_object.setup_opa_structure(pressure)
                        rt_object.calc_flux(temperature,
                                self.abundances,
                                1e1**self.params[self.key_gravity].value,
                                self.MMW,
                                contribution=contribution,
                                )
                        f_lambda.append(rt_object.flux*rt_object.freq**2./nc.c * 1e-7)
                    f_lambda = np.average(f_lambda, axis=0, weights=[1.-f_spot, f_spot])
                else:
                    rt_object.setup_opa_structure(self.press)
                    rt_object.calc_flux(self.temp,
                            self.abundances,
                            1e1**self.params[self.key_gravity].value,
                            self.MMW,
                            contribution=contribution,
                            )
                    # convert flux f_nu to f_lambda in unit of W/cm**2/um
                    f_lambda = rt_object.flux*rt_object.freq**2./nc.c * 1e-7
                wlen_nm = nc.c/rt_object.freq/1e-7
                if self.key_radius in self.params:
                    f_lambda *= (self.params[self.key_radius].value * nc.r_jup \
                                / self.params[self.key_distance].value / nc.pc)**2
                model.append([wlen_nm, f_lambda])
            self.model_native[instrument] = model

        if contribution:
            contr_em = rt_object.contr_em
            contr = np.zeros(contr_em.shape[0])
            for i,h in enumerate(contr_em):
                integrand1 = wlen_nm*h*f_lambda
                integrand2 = wlen_nm*f_lambda
                integral1 = np.trapz(integrand1, wlen_nm)
                integral2 = np.trapz(integrand2, wlen_nm)
                contr[i] = integral1/integral2
            return contr


    def add_telluric_model(self, 
                           tellu_species=['H2O', 'CH4', 'CO2'],
                           grid_path='../',
                           ):
        if self.fit_telluric:
        
            self.tellu_species = tellu_species

            tellu_grid = TelluricGrid(
                            os.path.join(self.out_dir, grid_path),
                            free_species=self.tellu_species)
            
            self.tellu_grid = tellu_grid.grid
            self.humidity_range = tellu_grid.humidity_range
            self.ppmv_range = tellu_grid.ppmv_range
            self.temp_range = tellu_grid.temp_range
            self.fixed_species = [s for s in tellu_grid.all_species if s not in tellu_grid.free_species]
            
            self.add_parameter(self.key_airmass, prior=(1., 3.))
            self.add_parameter(self.key_tellu_temp, 
                               prior=(self.temp_range[0], 
                                      self.temp_range[-1]))
            
            for species in tellu_species:
                param_name = "tellu_" + species.split('_')[0]
                if species == 'H2O':
                    self.add_parameter(param_name, 
                                       prior=(self.humidity_range[0], 
                                              self.humidity_range[-1]))
                else:
                    self.add_parameter(param_name, 
                                       prior=(self.ppmv_range[0], 
                                              self.ppmv_range[-1]))


    def forward_model_telluric(self):

        y = np.ones_like(self.tellu_grid['WAVE'])
        for species in self.tellu_species:
            param_name = "tellu_" + species.split('_')[0]
            if species == 'H2O':
                rel_range = self.humidity_range
            else:
                rel_range = self.ppmv_range
            y *= RegularGridInterpolator((self.temp_range, rel_range), 
                                self.tellu_grid[species], 
                                bounds_error=False, fill_value=None)(
                                [self.params[self.key_tellu_temp].value, 
                                 self.params[param_name].value])[0]
            y[y<0.] = 0.
        for species in self.fixed_species:
            y *= self.tellu_grid[species]
        tellu_native = y**(self.params[self.key_airmass].value)
        self.model_tellu = interp1d(self.tellu_grid['WAVE'], tellu_native, 
                                    bounds_error=False, fill_value='extrapolate')
        
        # if self.debug:
        #     plt.plot(self.tellu_grid['WAVE'], tellu_native)
        #     plt.show()

    def plot_model_debug(self, model):
        for instrument in self.obs.keys():
            for dt in model[instrument]:
                wave_tmp, flux_tmp = dt[0], dt[1]
                plt.plot(wave_tmp, flux_tmp, 'k')
        plt.show()


    def plot_rebin_model_debug(self, model):
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3,1)})
        
        for instrument in self.obs.keys():
            for i, flux_tmp in enumerate(model[instrument]):
                if instrument == 'photometry':
                    wave_tmp = self.obs[instrument][i].wlen.mean()
                    flux_obs = self.obs[instrument][i].f_lambda
                    err_obs = self.obs[instrument][i].f_err
                    # print(flux_obs, flux_tmp)
                else:
                    wave_tmp = self.obs[instrument].wlen[i]
                    flux_obs = self.obs[instrument].flux[i]
                    err_obs = self.obs[instrument].err[i]
                    axes[0].errorbar(wave_tmp, flux_obs, err_obs, color='r', alpha=0.8)
                    axes[0].plot(wave_tmp, flux_tmp, color='k', alpha=0.8, zorder=10)
                    axes[1].plot(wave_tmp, (flux_obs-flux_tmp), color='k', alpha=0.8, zorder=10)
        # axes[0].set_ylim(0.1, 1.2)
        # axes[1].set_ylim(-5,5)
        axes[0].set_ylabel(r'Flux')
        axes[1].set_ylabel(r'Residual')
        axes[1].set_xlabel('Wavelength (nm)')
        plt.show()
        plt.close(fig)

    

    def apply_rot_broaden_rv_shift(self):
        if self.key_limb_dark_u in self.params:
            self.limb = self.params[self.key_limb_dark_u].value
        self.model_spin = {}
        for instrument in self.obs.keys():
            model = self.model_native[instrument]
            model_tmp = []
            for dt in model:
                wave_tmp, flux_tmp = dt[0], dt[1]
                wave_shift = wave_tmp * (1. + self.params["vsys"].value*1e5 / nc.c) 
                wlen_up = np.linspace(wave_tmp[0], wave_tmp[-1], len(wave_tmp)*20)

                flux_take = interp1d(wave_shift, flux_tmp, bounds_error=False, fill_value='extrapolate')(wlen_up)
                flux_spin = pyasl.fastRotBroad(wlen_up, flux_take, self.limb, self.params[self.key_spin].value)
                model_tmp.append([wlen_up, flux_spin])
            self.model_spin[instrument] = model_tmp
        
        

    def add_instrument_kernel(self, Lorentzian_kernel=False):
        if self.fit_instrument_kernel:
            for instrument in self.obs.keys():
                self.add_parameter(f'{instrument}_G', prior=(0.3e5, 2e5))
                if Lorentzian_kernel:
                    self.add_parameter(f'{instrument}_L', prior=(0.1, 3))


    def apply_instrument_broaden(self):
        self.model_convolved = {}
        for instrument in self.obs.keys():
            if instrument == 'photometry':
                # skip broadening for photometry objects 
                self.model_convolved[instrument] = self.model_spin[instrument]
            else:
                if f'{instrument}_G' in self.params:
                    inst_G = self.params[f'{instrument}_G'].value
                else:
                    inst_G = self.obs[instrument].R
                if f'{instrument}_L' in self.params:
                    inst_L = self.params[f'{instrument}_L'].value
                else:
                    inst_L = 0
                model_target = self.model_spin[instrument]
                model_tmp = []
                for dt in model_target:
                    wave_tmp, flux_tmp = dt[0], dt[1]
                    flux_full = flux_tmp * self.model_tellu(wave_tmp)
                    flux_conv = su.SpecConvolve_GL(wave_tmp, flux_full, inst_G, inst_L)
                    model_tmp.append([wave_tmp, flux_conv])
                self.model_convolved[instrument] = model_tmp


    def apply_rebin_to_obs_wlen(self):
        self.model_rebin = {}
        for instrument in self.obs.keys():
            model_tmp = []
            if instrument == 'photometry':
                for model_target, obs_target in zip(self.model_convolved[instrument], self.obs[instrument]):
                    flux_rebin = rgw.rebin_give_width(
                                    model_target[0], 
                                    model_target[1],
                                    obs_target.wlen, 
                                    obs_target.wlen_bins,
                                    )
                    integrand1 = obs_target.wlen*obs_target.flux*flux_rebin
                    integrand2 = obs_target.wlen*obs_target.flux
                    integral1 = np.trapz(integrand1, obs_target.wlen)
                    integral2 = np.trapz(integrand2, obs_target.wlen)
                    model_tmp.append(integral1/integral2)
                    
            else:
                model_target = self.model_convolved[instrument]
                obs_target = self.obs[instrument]
                for i in range(obs_target.wlen.shape[0]):
                    flux_rebin = rgw.rebin_give_width(
                            model_target[i//obs_target.detector_bin][0], 
                            model_target[i//obs_target.detector_bin][1],
                            obs_target.wlen[i], 
                            obs_target.wlen_bins[i],
                            )
                    model_tmp.append(flux_rebin)
            self.model_rebin[instrument] = np.array(model_tmp)

        # if self.debug:
            # self.plot_model_debug(self.model_native)
            # self.plot_model_debug(self.model_spin)
            # self.plot_model_debug(self.model_convolved)
            # self.plot_rebin_model_debug(self.model_rebin)

    def add_poly_model(self): 
        if self.fit_poly:
            for instrument in self.obs.keys():
                if instrument != 'photometry':
                    for i in range(self.obs[instrument].Nchip):
                        for o in range(1, self.fit_poly+1):
                            self.add_parameter(f'poly_{instrument}_{o}_{i:02}', prior=(-5e-2/o, 5e-2/o))
        

    def apply_poly_continuum(self):
        for instrument in self.obs.keys():
            if instrument != 'photometry':
                model_target = self.model_rebin[instrument]
                obs_target = self.obs[instrument]
                model_tmp = []
                for i, y_model in enumerate(model_target):
                    x = obs_target.wlen[i]

                    if self.fit_poly:
                        # correct for the slope or higher order poly of the continuum
                        poly = [1.] 
                        for o in range(1, self.fit_poly+1):
                            poly.append(self.params[f'poly_{instrument}_{o}_{i:02}'].value)
                        y_poly = Chev.chebval((x - np.mean(x))/(np.mean(x)-x[0]), poly)
                        y_model *= y_poly

                    model_tmp.append(y_model)
                self.model_rebin[instrument] = np.array(model_tmp)

        # if self.debug:
        #     self.plot_rebin_model_debug(self.model_rebin)


    def add_GP(self, mu_local_GP=None, dmu=0.3,
               GP_chip_bin=None, 
               prior_amp_global=(-4,0), prior_tau_global=(-1,1),
               prior_amp_local=(-4,0), prior_tau_local=(-2,0.),
               ):
        if self.fit_GP:
            for instrument in self.obs.keys():
                if instrument != 'photometry':
                    if GP_chip_bin is None: #use one kernel for all orders
                        self.add_parameter(f"GP_{instrument}_amp", prior=prior_amp_global)
                        self.add_parameter(f"GP_{instrument}_tau", prior=prior_tau_global)
                    else: #different kernel for each chip
                        for i in range(0, self.obs[instrument].Nchip, GP_chip_bin):
                            self.add_parameter(f"GP_{instrument}_amp_{i:02}", prior=prior_amp_global)
                            self.add_parameter(f"GP_{instrument}_tau_{i:02}", prior=prior_tau_global)
            if mu_local_GP is not None:
                for k, mu in enumerate(mu_local_GP):
                    self.add_parameter(f"muloc_{k:02}", prior=(mu-dmu, mu+dmu))
                    self.add_parameter(f"amploc_{k:02}", prior=prior_amp_local)
                    self.add_parameter(f"sigloc_{k:02}", prior=prior_tau_local)

    def calc_covariance(self, obs_target):
        amp = [1e1**self.params[key].value for key in self.params \
                                        if "amp" in key.split("_")]
        tau = [1e1**self.params[key].value for key in self.params \
                            if "tau" in key.split("_")]
        
        obs_target.make_covariance(amp, tau)
        
        amp_loc = [1e1**self.params[key].value for key in self.params \
                            if "amploc" in key.split("_")]
        sigma_loc = [1e1**self.params[key].value for key in self.params \
                            if "sigloc" in key.split("_")]
        mu_loc = [self.params[key].value for key in self.params \
                            if "muloc" in key.split("_")]

        if mu_loc:
            obs_target.make_covariance_local(amp_loc, mu_loc, sigma_loc)
        
        cov = obs_target.cov
        return cov


    def calc_scaling(self, y_model, y_data, y_cov, rcond=None):
        if y_cov.ndim == 2:
            # sparse Cholesky decomposition
            cov_chol = cholesky(y_cov)
            # Scale the model flux to minimize the chi-squared
            lhs = np.dot(y_model.T, cov_chol.solve_A(y_model))
            rhs = np.dot(y_model.T, cov_chol.solve_A(y_data))
            if y_model.ndim == 1:
                f_det = rhs/lhs
            else:
                f_det, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
            
        elif y_cov.ndim == 1:
            # Scale the model flux to minimize the chi-squared
            lhs = np.dot(y_model.T, 1/y_cov * y_model)
            rhs = np.dot(y_model.T, 1/y_cov * y_data)
            if y_model.ndim == 1:
                f_det = rhs/lhs
            else:
                f_det, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
            
        return f_det.T
    
    def calc_err_inflation(self, y_model, y_data, y_cov):
        if y_cov.ndim == 2:
            cov_chol = cholesky(y_cov)
            chi_squared = np.dot((y_data-y_model), cov_chol.solve_A(y_data-y_model))
        elif y_cov.ndim == 1:
            chi_squared = np.sum(((y_data-y_model)**2/y_cov))
        return np.sqrt(chi_squared/len(y_data))

   
    def calc_logL(self, y_model, y_data, y_cov):
        if y_cov.ndim == 2:
            cov_chol = cholesky(y_cov)
            # Compute the chi-squared error
            chi_squared = np.dot((y_data-y_model), cov_chol.solve_A(y_data-y_model))

            # Log of the determinant (avoids under/overflow issues)
            logdet_cov = cov_chol.logdet()

        elif y_cov.ndim == 1:
            chi_squared = np.sum(((y_data-y_model)**2/y_cov))

            # Log of the determinant
            logdet_cov = np.sum(np.log(y_cov))
        # print(chi_squared, logdet_cov)

        return -(len(y_data)*np.log(2*np.pi)+chi_squared+logdet_cov)/2., chi_squared



    def prior(self, cube, ndim, nparams):
        pri=Prior()
        i = 0
        indices = []
        for key in self.params:
            if self.params[key].is_free:
                a, b = self.params[key].prior
                # cube[i] = a+(b-a)*cube[i]
                cube[i]=pri.GeneralPrior(cube[i], self.params[key].prior_type, a, b)
                # find indices of the free temperature parameters
                if key == self.key_t_bottom:
                    t_i = cube[i]
                elif key.split("_")[0] == 't':
                    indices.append(i)
                elif key == self.key_teff:
                    t_eff = cube[i]
                elif key == self.key_teff_spot:
                    indice_t_spot = i
                i += 1


        if self.key_t_bottom in self.params:
            # enforce decreasing temperatures from bottom to top layers 
            for k in indices:
                t_i = t_i * cube[k] #(1.-0.5*cube[k])
                cube[k] = t_i
        if self.key_teff_spot in self.params:
            cube[indice_t_spot] = t_eff * cube[indice_t_spot]


    def loglike(self, cube, ndim, nparams):
        log_likelihood, chi2_reduced = 0., 0.

        if self.leave_out is not None:
            model_reduced = []
            for neglect_sp in self.leave_out:
                self.set_parameter_values(cube)

                self.forward_model_pRT(leave_out=neglect_sp)
                if self.fit_telluric:
                    self.forward_model_telluric()
                self.apply_rot_broaden_rv_shift()
                self.apply_instrument_broaden()
                self.apply_rebin_to_obs_wlen()
                if self.fit_poly:
                    self.apply_poly_continuum()
                model_reduced.append(copy.copy(self.model_rebin))
            
        # draw parameter values from cube
        self.set_parameter_values(cube)

        self.forward_model_pRT()
        if self.fit_telluric:
            self.forward_model_telluric()
        self.apply_rot_broaden_rv_shift()
        self.apply_instrument_broaden()
        self.apply_rebin_to_obs_wlen()
        if self.fit_poly:
            self.apply_poly_continuum()


        self.flux_scaling, self.err_infaltion, self.model_single = {}, {}, {}
        for instrument in self.obs.keys():
            if instrument == 'photometry':
                for model_target, obs_target in zip(self.model_rebin[instrument], self.obs[instrument]):
                    chi2 = ((obs_target.f_lambda - model_target)/obs_target.f_err)**2 
                    log_det = np.log(2*np.pi*obs_target.f_err**2)
                    log_likelihood += -(chi2 + log_det)/2.
                    chi2_reduced += chi2
                    N += 1
            else:
                model_target = self.model_rebin[instrument]
                model_single = []
                if self.leave_out is not None:
                    for ind_sp in range(len(self.leave_out)):
                        model_single.append(model_target - model_reduced[ind_sp][instrument])

                obs_target = self.obs[instrument]._copy()
                if self.fit_GP:
                    cov = self.calc_covariance(obs_target)

                else:
                    cov = obs_target.err**2
                
                if self.fit_spline:
                    _, M_spline = obs_target.make_spline_model(self.fit_spline)
                    model_splined = M_spline * model_target[:, :, None]

                    f_dets = np.ones(model_splined.shape[:-1])
                    for i in range(obs_target.Nchip):
                        f_det = self.calc_scaling(model_splined[i], obs_target.flux[i], cov[i])
                        f_dets[i] = f_det
                        model_target[i] = np.dot(model_splined[i], f_det)
                        # if self.leave_out is not None:
                        #     model_single[i] = np.dot(model_single[i], f_det)

                if self.fit_scaling:
                    f_dets = np.ones(model_target.shape[:-1])
                    for i in range(obs_target.Nchip):
                        # plt.imshow(cov[i].toarray())
                        # plt.show()
                        f_det = self.calc_scaling(model_target[i], obs_target.flux[i], cov[i])
                        f_dets[i] = f_det
                        model_target[i] *= f_det
                        if self.leave_out is not None:
                            for ind_sp in range(len(self.leave_out)):
                                model_single[ind_sp][i] *= f_det

                #                 plt.plot(obs_target.wlen[i], model_single[ind_sp][i], alpha=0.8,color=cmap(ind_sp))
                #         plt.plot(obs_target.wlen[i], model_target[i], alpha=0.8,color='k')
                # plt.show()

                # add model uncertainties due to the inacurate molecular line list
                if "beta_0" in self.params:
                    for ind_sp in range(len(self.leave_out)):
                        key = f"beta_{ind_sp}"
                        cov += (self.params[key].value * model_single[ind_sp])**2
                if self.key_teff_tolerance in self.params:
                    cov += 1e1**self.params[self.key_teff_tolerance].value #* self.temp[np.searchsorted(self.press, 1.)] / 3000.


                betas = np.ones(obs_target.Nchip)
                if self.fit_err_inflation:
                    for i in range(obs_target.Nchip):
                        beta = self.calc_err_inflation(model_target[i], obs_target.flux[i], cov[i])
                        cov[i] *= beta**2
                        betas[i] = beta
                
                # Add to the log-likelihood
                for i in range(obs_target.Nchip):
                    log_l, chi2 = self.calc_logL(model_target[i], obs_target.flux[i], cov[i])
                    log_likelihood += log_l
                    chi2_reduced += chi2/obs_target.flux.size
            
                self.model_rebin[instrument] = model_target
                self.flux_scaling[instrument] = f_dets
                self.err_infaltion[instrument] = betas
                if self.leave_out is not None:
                    self.model_single[instrument] = model_single
        
        self.ln_L = log_likelihood + self.ln_L_penalty

        if self.debug:
            print("Chi2_r: ", chi2_reduced)
            print(self.ln_L, self.ln_L_penalty, log_likelihood)
            print(self.flux_scaling, self.err_infaltion)
            # self.plot_rebin_model_debug(self.model_rebin)
            # self.plot_rebin_model_debug(self.model_reduce)
        
        return self.ln_L


    def setup(self, 
              obs,
              line_species,
              param_prior={},
              press=None,
              N_t_knots=None, 
              PT_profile='free',
              chemistry='free',
              line_species_ck=None,
              fit_instrument_kernel=True,
              Lorentzian_kernel=False,
              leave_out=None,
              fit_GP=False, 
              fit_poly=1, 
              fit_scaling=True, 
              fit_spline=False,
              fit_err_inflation=True,
              fit_telluric=False, 
              tellu_grid_path='../',
              stellar_grid_path=None,
              PT_penalty_order=0,
              mu_local_GP=None,
              inhomogeneous=False,
              ):

        if press is None:
            self.press = np.logspace(-4,2,50)
        else:
            self.press = press

        self.line_species = line_species
        if line_species_ck is not None:
            self.line_species_ck = line_species_ck
        else:
            self.line_species_ck = line_species
        self.fit_GP = fit_GP
        self.fit_poly = fit_poly
        self.fit_scaling = fit_scaling
        self.fit_spline = fit_spline
        self.fit_err_inflation = fit_err_inflation
        self.fit_telluric = fit_telluric
        self.fit_instrument_kernel = fit_instrument_kernel
        self.leave_out = leave_out
        self.PT_penalty_order = PT_penalty_order

        self.add_observation(obs)
        assert self.obs, "No input observations provided"

        print("Creating pRT objects for input data...")
        self.add_pRT_objects()

        if PT_profile == 'free':
            self.add_free_PT_model(N_t_knots)
        elif PT_profile == 'grid':
            self.add_grid_PT_model(inhomo=inhomogeneous,
                                   grid_path=stellar_grid_path)


        if chemistry == 'free':
            self.add_free_chem_model()
        elif chemistry == 'equ':
            self.add_equ_chem_model()
        
        self.add_telluric_model(grid_path=tellu_grid_path)
        
        self.add_instrument_kernel(Lorentzian_kernel)
        self.add_poly_model()
        self.add_GP(mu_local_GP=mu_local_GP)
        
        self.set_parameter_priors(param_prior)

        # save parameter names and count number of params
        parameters = []
        for x in self.params:
            if self.params[x].is_free:
                parameters.append(self.params[x].name)
        self.n_params = len(parameters)
        print(f"{self.n_params} free parameters in total.")
        json.dump(parameters, open(self.prefix + 'params.json', 'w'))


    def set_parameter_priors(self, param_prior):
        for key in param_prior:
            if key in self.params:
                if isinstance(param_prior[key], (float, int, type(None))):
                    self.params[key].set_value(param_prior[key])
                    self.params[key].is_free = False
                else:
                    self.params[key].set_prior(param_prior[key])
            else:
                if isinstance(param_prior[key], (float, int, type(None))):
                    self.add_parameter(key, value=param_prior[key], is_free=False)
                else:
                    self.add_parameter(key, prior=param_prior[key])

    def set_parameter_values(self, params):
        # set parameters
        i_p = 0 # parameter count
        for pp in self.params:
            if self.params[pp].is_free:
                self.params[pp].set_value(params[i_p])
                i_p += 1
                if self.debug:
                    print(f"{i_p} \t {pp} \t {self.params[pp].value}")


    def run(self, n_live_points=500, debug=False):

        self.debug = debug

        pymultinest.run(self.loglike,
            self.prior,
            self.n_params,
            outputfiles_basename=self.prefix,
            resume = False, 
            verbose = True, 
            const_efficiency_mode = True, 
            sampling_efficiency = 0.05,
            n_live_points = n_live_points)
        

    def evaluation(self, 
                   quantiles = [0.16,0.5,0.84], #[0.05,0.5,0.95],
                   corner_plot=False, 
                   params_corner=None,
                   which_best='mean', 
                   ):
        self.debug = True
        self.main_color = 'C1'

        parameters = json.load(open(self.prefix + 'params.json'))

        analyzer = pymultinest.Analyzer(n_params=self.n_params, outputfiles_basename=self.prefix)
        s = analyzer.get_stats()
        
        samples = np.genfromtxt(self.prefix+'post_equal_weights.dat')

        # add derived C/O and 12/13C ratio if not in the parameter list
        if self.key_carbon_to_oxygen not in parameters:
            c2h = calc_abundance_ratio_posterior('C','H', parameters, samples)
            samples = np.concatenate((c2h[:,np.newaxis], samples), axis=1)
            parameters = [self.key_metallicity] + parameters
            c2o = calc_abundance_ratio_posterior('C','O', parameters, samples)
            samples = np.concatenate((c2o[:,np.newaxis], samples), axis=1)
            parameters = [self.key_carbon_to_oxygen] + parameters
        if self.key_carbon_iso not in parameters:
            c_iso = calc_abundance_ratio_posterior('C','13C', parameters, samples)
            samples = np.concatenate((c_iso[:,np.newaxis], samples), axis=1)
            parameters = [self.key_carbon_iso] + parameters
            
        
        json.dump(s, open(self.prefix + 'stats.json', 'w'), indent=4)

        with open(self.prefix+'summary.txt', 'w') as f:
            print('  marginal likelihood:', file=f)
            print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']), file=f)
            print(' ', file=f)
            print('  parameters \t values +- 1sigma:', file=f)
            param_quantiles = []
            for k, x in enumerate(samples[:,:-1].T):
                qs = self._quantile(x, quantiles)
                param_quantiles.append(qs)
                med = qs[1]
                sigma = (qs[-1]-qs[0])/2.
                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
                print(fmts % (parameters[k], med, sigma), file=f)

        if params_corner is not None:
            param_indices = []
            for i, p in enumerate(parameters):
                if p in params_corner:
                    param_indices.append(i)
        else:
            param_indices = range(len(parameters))
        samples_use = samples[:, param_indices]
        param_labels = [parameters[i] for i in param_indices]
        param_quantiles = np.array(param_quantiles)[param_indices]
        param_range = [(4*(qs[0]-qs[1])+qs[1], 4*(qs[2]-qs[1])+qs[1]) for qs in param_quantiles] 

        # corner plot
        if corner_plot:

            fig = plt.figure(figsize=(15,15))
            fig = corner.corner(samples_use, 
                                fig=fig, 
                                labels=param_labels,
                                range=param_range,
                                show_titles=True, 
                                title_fmt='.2f', 
                                # use_math_text=True, 
                                title_kwargs={'fontsize':9}, 
                                # bins=20, 
                                max_n_ticks=4, 
                                quantiles=quantiles, 
                                color=self.main_color, 
                                hist_kwargs={'color':self.main_color}, 
                                fill_contours=True, 
                                )

            param_envelope, envelope_temp, envelope_vmr = [], [], []
            for m in s['marginals']:
                param_p = []
                for k in range(1,4):
                    lo, hi = m[f'{k}sigma']
                    param_p.append(lo)
                    param_p.append(hi)
                param_p.append(m['median'])
                param_envelope.append(param_p)
            param_envelope = np.array(param_envelope).T
            for param_set in param_envelope:
                # set parameters
                self.set_parameter_values(param_set)
                self.PT_model()
                self.abundances, self.MMW = self.chem_model()
                envelope_temp.append(self.temp)
                envelope_vmr.append(self.abundances)
            
            # plot the PT profile and envelope
            ax_PT = fig.add_axes([0.52,0.7,0.2,0.2])
            self.make_PT_plot(ax_PT, envelope_temp)

            # Plot the VMR per species
            ax_vmr = fig.add_axes([0.78,0.7,0.2,0.2])
            self.make_VMR_plot(ax_vmr, envelope_vmr)

            plt.savefig(self.prefix+'corner.pdf')
            plt.close(fig)

        best_fit_params = s['modes'][0][which_best]
        self.loglike(best_fit_params[:self.n_params], self.n_params, self.n_params)
        # plot and save best-fit model
        for instrument in self.obs.keys():
            if instrument != 'photometry':
                self.make_best_fit_plot(self.obs[instrument], 
                                        self.model_rebin[instrument],
                                        self.prefix + f'{instrument}_best_fit_spec.pdf')
                model = SPEC2D(self.obs[instrument].wlen, self.model_rebin[instrument])
                model.save_spec1d(self.prefix + f'{instrument}_best_fit_spec.dat')
        
        best_fit_params = s['modes'][0][which_best]
        self.leave_out = ['H2O','CO_36']
        self.loglike(best_fit_params[:self.n_params], self.n_params, self.n_params)
        # plot ccf of residuals 
        for instrument in self.obs.keys():
            if instrument != 'photometry':
                self.make_ccf_plot(self.obs[instrument], self.model_rebin[instrument], 
                                    self.model_single[instrument],
                                    self.prefix + f'{instrument}_res_ccf.pdf')


    def average_post_model(self, N, instrument='crires'):
        samples = np.genfromtxt(self.prefix+'post_equal_weights.dat')
        indices = np.random.choice(range(samples.shape[0]), N)
        samples_use = samples[indices,:-1]
        sample_model = []

        for param_set in samples_use:
            self.loglike(param_set, self.n_params, self.n_params)
            sample_model.append(self.model_rebin[instrument])

        avg_model = np.mean(sample_model, axis=0)
        model_object = SPEC2D(self.obs[instrument].wlen, avg_model)
        self.obs[instrument].outliers(model_object)
        # residual.plot_spec1d('comb_model.pdf')
        


    def make_PT_plot(self, ax_PT, envelope_temp):
        for i in range(3):
            ax_PT.fill_betweenx(self.press, envelope_temp[i*2], envelope_temp[i*2+1],
                                facecolor=cmap_hue(i), zorder=10-i)
        ax_PT.plot(envelope_temp[-1], self.press, color=self.main_color, zorder=11)
        ax_PT.set_xlabel('Temperature (K)')
        ax_PT.set_ylabel('Pressure (bar)')
        ax_PT.set_yscale('log')
        ax_PT.set_ylim([self.press[-1], self.press[0]])
        ax_PT.set_xlim([envelope_temp[-3][0], envelope_temp[-2][-1]])

        contri = self.forward_model_pRT(contribution=True)
        ax_contri = ax_PT.twiny()
        ax_contri.set_yscale('log')
        ax_contri.plot(contri, self.press, ':k', zorder=-11)
        ax_contri.set_xlim(0,np.max(contri))
        ax_contri.tick_params(axis='x', which='both', top=False, labeltop=False)


    def make_VMR_plot(self, ax_vmr, envelope_vmr):
        for k, key in enumerate(self.line_species):
            for i in range(3):
                ax_vmr.fill_betweenx(self.press, 
                                    envelope_vmr[i*2][key]/getMM(key)*self.MMW, 
                                    envelope_vmr[i*2+1][key]/getMM(key)*self.MMW,
                                    facecolor=cmap_hue(k*4+i), zorder=10-i)
            ax_vmr.plot(envelope_vmr[-1][key]/getMM(key)*self.MMW, self.press, color=cmap(k), label=key, zorder=11)
        ax_vmr.set_xlabel('VMR')
        ax_vmr.set_ylabel('Pressure (bar)')
        ax_vmr.set_yscale('log')
        ax_vmr.set_xscale('log')
        ax_vmr.set_ylim([self.press[-1], self.press[0]])
        ax_vmr.set_xlim([1e-8, 1e-1])
        ax_vmr.legend(loc='best')


    def make_best_fit_plot(self, obs, models, savename, labels=['model']):
        self._set_plot_style()
        nrows = obs.Nchip//3
        fig, axes = plt.subplots(nrows=nrows*2, ncols=1, sharex=True,
                          figsize=(12,nrows*3), constrained_layout=True,
                          gridspec_kw={"height_ratios": [3,1]*nrows})

        for i in range(nrows):
            ax, ax_res = axes[2*i], axes[2*i+1]
            wmin, wmax = obs.wlen[i*3][0], obs.wlen[min(i*3+2, obs.wlen.shape[0]-1)][-1]
            ymin, ymax = 1, 0
            if self.fit_GP:
                covs = self.calc_covariance(obs)

            for j in range(min(3, obs.wlen.shape[0]-3*i)):
                x, y, y_err = obs.wlen[i*3+j], obs.flux[i*3+j], obs.err[i*3+j]
                if self.fit_GP:
                    cov = covs[i*3+j].toarray()
                    mask = (cov==0.)
                    y_err = np.array([np.sqrt(np.std(cov[r][~mask[r]])) for r in range(len(cov))])

                ax.errorbar(x, y, y_err, color='k', alpha=0.8, capsize=0.5)
                ax_res.fill_between(x, -3*y_err, 3*y_err, color=cmap(7), alpha=0.2)
                ax_res.fill_between(x, -y_err, y_err, color=cmap(7), alpha=0.4)


                if not isinstance(models, list):
                    models = [models]
                for k, model in enumerate(models):
                    y_model = model[i*3+j]
                    ax.plot(x, y_model,  color=cmap(k), 
                            label=labels[k],
                            alpha=0.8, zorder=10)
                    ax_res.plot(x, y-y_model,  color=cmap(k), alpha=0.8)
                nans = np.isnan(y)
                vmin, vmax = np.percentile(y_model, (1, 99))
                ymin, ymax = min(vmin, ymin), max(vmax, ymax)
                rmin, rmax = np.percentile((y-y_model)[~nans], (0.1, 99.9))

            ax.set_xlim((wmin, wmax))
            ax_res.set_xlim((wmin, wmax))
            ax.set_ylim((ymin*0.9, ymax*1.1))
            ax_res.set_ylim((rmin*0.9, rmax*1.1))
            ax.set_xticklabels([])
            ax.set_ylabel(r'Flux')
            ax_res.set_ylabel(r'Residual')
        axes[0].legend()
        axes[-1].set_xlabel('Wavelength (nm)')
        plt.savefig(savename)
        # plt.show()
        plt.close(fig)


    def make_ccf_plot(self, obs, model, model_single, savename):
        self._set_plot_style()
        nrows = obs.Nchip//3
        fig, axes = plt.subplots(nrows=nrows, ncols=2, 
                          figsize=(14,nrows*3), constrained_layout=True,
                          gridspec_kw={"width_ratios": [5,1]},
                          )
        if nrows==1:
            axes = axes[np.newaxis,:]

        for k, y_singles in enumerate(model_single):
            ccf = []
            for i in range(nrows):
                wmin, wmax = obs.wlen[i*3][0], obs.wlen[min(i*3+2, obs.Nchip-1)][-1]

                x, y = obs.wlen[i*3:i*3+3].flatten(), obs.flux[i*3:i*3+3].flatten()
                # y_err = obs.err[i*3:i*3+3].flatten()
                y_model, y_single = model[i*3:i*3+3].flatten(), y_singles[i*3:i*3+3].flatten()
                v, ccf_order = su.CCF_doppler(x, y-y_model+y_single, 
                                              x, y_single-np.mean(y_single), 
                                              800, 1)
                ccf.append(ccf_order)
                in_range = (v > -400) & (v<-400)
                if self.leave_out[k] == 'CO_36':
                    for j in range(min(3, obs.Nchip-3*i)):
                        axes[i,0].plot(obs.wlen[i*3+j], 
                                       obs.flux[i*3+j]-model[i*3+j]+y_singles[i*3+j], 
                                       color='k', alpha=0.8)
                axes[i,0].plot(x, y_single, color=cmap(k), alpha=0.8)
                axes[i,1].plot(v, ccf_order/np.std(ccf_order[~in_range]), 
                               color=cmap(k), alpha=0.8, label=self.leave_out[k])
                axes[i,1].set_xlabel('Velocity (km/s)')
                axes[i,1].set_ylabel('CCF')

                nans = np.isnan(y)
                rmin, rmax = np.percentile((y-y_model)[~nans], (1, 99))

                axes[i,0].set_xlim((wmin, wmax))
                axes[i,0].set_ylim((rmin*1.5, rmax*1.5))
                axes[i,0].set_xlabel('Wavelength (nm)')
                axes[i,0].set_ylabel(r'Residual')
                axes[i,1].legend()

        plt.savefig(savename)
        plt.show()
        plt.close(fig)


    def make_ccf_plot_by_detector(self, obs, model, model_single, savename):
        self._set_plot_style()
        nrows = obs.Nchip//3
        fig = plt.figure(constrained_layout=True, figsize=(14,5))
        gs = fig.add_gridspec(nrows=nrows*2,ncols=4,height_ratios=[2,1]*nrows)
        ax_ccf_sum = fig.add_subplot(gs[:,-1])
        ax, ax_ccf = [], []
        for i in range(nrows):
            ax.append(fig.add_subplot(gs[2*i,:3]))
            for j in range(min(3, obs.Nchip-3*i)):
                ax_ccf.append(fig.add_subplot(gs[2*i+1,j]))

        ccf_species = []
        for k, y_singles in enumerate(model_single):
            ccf = []
            for i in range(nrows):
                wmin, wmax = obs.wlen[i*3][0], obs.wlen[min(i*3+2, obs.Nchip-1)][-1]

                for j in range(min(3, obs.Nchip-3*i)):
                    x, y, y_err = obs.wlen[i*3+j], obs.flux[i*3+j], obs.err[i*3+j]
                    y_model, y_single = model[i*3+j], y_singles[i*3+j]
                    v, ccf_order = su.CCF_doppler(x, y-y_model+y_single, x, y_single-np.mean(y_single), 1000, 1)
                    ccf.append(ccf_order)
                    in_range = (v > -500) & (v<-500)
                    if self.leave_out[k] == 'CO_36':
                        ax[i].plot(x, y-y_model+y_single, color='k', alpha=0.8)
                    ax[i].plot(x, y_single, color=cmap(k), alpha=0.8)
                    ax_ccf[i*3+j].plot(v, ccf_order/np.std(ccf_order[~in_range]), color=cmap(k), alpha=0.8)
                    ax_ccf[i*3+j].set_xlabel('Velocity (km/s)')
                    ax_ccf[i*3+j].set_ylabel('CCF')


                    nans = np.isnan(y)
                    rmin, rmax = np.percentile((y-y_model)[~nans], (1, 99))

                ax[i].set_xlim((wmin, wmax))
                ax[i].set_ylim((rmin*1.5, rmax*1.5))
                ax[i].set_xlabel('Wavelength (nm)')
                ax[i].set_ylabel(r'Residual')

            ccf_species= np.sum(ccf, axis=0)
            ax_ccf_sum.plot(v, ccf_species/np.std(ccf_species[~in_range]), label=self.leave_out[k])
        ax_ccf_sum.legend()

        plt.savefig(savename)
        plt.show()
        plt.close(fig)


    def _set_plot_style(self):
        plt.rcParams.update(plt.rcParamsDefault)
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
            "lines.linewidth": 1.,   
            'image.origin': 'lower',
            'image.cmap': 'cividis',
            "savefig.dpi": 300,   
            })

    def _quantile(self, x, q, weights=None):
        """
        Compute (weighted) quantiles from an input set of samples.
        Parameters
        ----------
        x : `~numpy.ndarray` with shape (nsamps,)
            Input samples.
        q : `~numpy.ndarray` with shape (nquantiles,)
        The list of quantiles to compute from `[0., 1.]`.
        weights : `~numpy.ndarray` with shape (nsamps,), optional
            The associated weight from each sample.
        Returns
        -------
        quantiles : `~numpy.ndarray` with shape (nquantiles,)
            The weighted sample quantiles computed at `q`.
        """

        # Initial check.
        x = np.atleast_1d(x)
        q = np.atleast_1d(q)

        # Quantile check.
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("Quantiles must be between 0. and 1.")

        if weights is None:
            # If no weights provided, this simply calls `np.percentile`.
            return np.percentile(x, list(100.0 * q))
        else:
            # If weights are provided, compute the weighted quantiles.
            weights = np.atleast_1d(weights)
            if len(x) != len(weights):
                raise ValueError("Dimension mismatch: len(weights) != len(x).")
            idx = np.argsort(x)  # sort samples
            sw = weights[idx]  # sort weights
            cdf = np.cumsum(sw)[:-1]  # compute CDF
            cdf /= cdf[-1]  # normalize CDF
            cdf = np.append(0, cdf)  # ensure proper span
            quantiles = np.interp(q, cdf, x[idx]).tolist()
            return quantiles
        
