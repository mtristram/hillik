import os
import time
from itertools import combinations_with_replacement as cwr
from typing import Sequence, Union

import astropy.units as u
import h5py
import numpy as np

from cobaya.log import LoggedError
from cobaya.theory import Theory
from scipy import constants
from scipy import integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

#from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy.special import sici
from numpy import sin, cos

verbose_timing = True
save_outputs = False

######## MD
# check *np.log(10) in tSZ and CIB integ of dn_dm
# check z=1e-4 makes interpolate P(k) at k>500
# amplitude =/= MD
# amplitude / shape =/=Abi stand alone
# cc MjKcm-1 fnu ok

## -> 3 big changes:
## * z <0.012
## * more masse bins up to 1e15
## * P0 6.41 -> 6.41 * h*(-3/2)


# fmt: off
_default_ell_sampling = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 114, 187, 200, 320, 502, 684, 890, 1158, 1505, 1956, 3000, 5000, 8000, 10000, 13499])
#lmax = 13500
#_default_ell_sampling = np.arange(2,lmax, 200)




class Tinker08:
    r"""This is the function giving f(sigma)

    All the parameters mentioned here are taken for \Delta = 200 z is the redshift and redshift
    evolution has to be considered here It has to be noted that values provided in Tinker 2008 paper
    are wrt to mean background density. If we want to calculate the best fit parameter values wrt to
    critical background density, we have to change the corresponding delta_halo value and
    interpolate the values for the corresponding delta_halo values.

    See https://arxiv.org/abs/0803.2706 for details

    """

    def __init__(self):
        self.delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        self.a = {  # -- A
            "A_200": 1.858659e-01,
            "A_300": 1.995973e-01,
            "A_400": 2.115659e-01,
            "A_600": 2.184113e-01,
            "A_800": 2.480968e-01,
            "A_1200": 2.546053e-01,
            "A_1600": 2.600000e-01,
            "A_2400": 2.600000e-01,
            "A_3200": 2.600000e-01,
            # -- a
            "a_200": 1.466904,
            "a_300": 1.521782,
            "a_400": 1.559186,
            "a_600": 1.614585,
            "a_800": 1.869936,
            "a_1200": 2.128056,
            "a_1600": 2.301275,
            "a_2400": 2.529241,
            "a_3200": 2.661983,
            # --- b
            "b_200": 2.571104,
            "b_300": 2.254217,
            "b_400": 2.048674,
            "b_600": 1.869559,
            "b_800": 1.588649,
            "b_1200": 1.507134,
            "b_1600": 1.464374,
            "b_2400": 1.436827,
            "b_3200": 1.405210,
            # --- c
            "c_200": 1.193958,
            "c_300": 1.270316,
            "c_400": 1.335191,
            "c_600": 1.446266,
            "c_800": 1.581345,
            "c_1200": 1.795050,
            "c_1600": 1.965613,
            "c_2400": 2.237466,
            "c_3200": 2.439729,
        }

        A_array = np.array([self.a["A_%s" % d] for d in self.delta_virs])
        a_array = np.array([self.a["a_%s" % d] for d in self.delta_virs])
        b_array = np.array([self.a["b_%s" % d] for d in self.delta_virs])
        c_array = np.array([self.a["c_%s" % d] for d in self.delta_virs])

        self.A_0 = InterpolatedUnivariateSpline(self.delta_virs, A_array)
        self.a_0 = InterpolatedUnivariateSpline(self.delta_virs, a_array)
        self.b_0 = InterpolatedUnivariateSpline(self.delta_virs, b_array)
        self.c_0 = InterpolatedUnivariateSpline(self.delta_virs, c_array)

    def fsigma(self, z, sigma, dhalo):
        # If dhalo is outside [200, 3200] may be the extrapolation is not ideal ?
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html
        # Maybe something like A_0[dhalo < 200] = a["A_200"] and A_0[dhalo > 3200] = a["A_3200"]
        A_0 = self.A_0(dhalo)
        a_0 = self.a_0(dhalo)
        b_0 = self.b_0(dhalo)
        c_0 = self.c_0(dhalo)

        A_exp = -0.14
        a_exp = -0.06
        A = A_0 * (1 + z) ** A_exp
        a = a_0 * (1 + z) ** a_exp
        alpha = 10 ** (-((0.75 / np.log10(dhalo / 75.0)) ** 1.2))
        b = b_0 * (1 + z) ** (-alpha)
        return A * ((sigma / b) ** -a + 1) * np.exp(-c_0 / sigma ** 2)


class HaloModel(Theory):

    KC = 1.0e-10  # Kennicutt constant for Chabroer IMF
    fsub = 0.134*np.log(10)  # fraction of the total halo mass in form of subhalos

    # Options for Pk.
    # Default options can be set globally, and updated from requirements as needed
    kmax: float = 10  # Maximum k (1/Mpc units) for Pk, or zero if not needed
    nonlinear: bool = False  # whether to get non-linear Pk from CAMB/Class
    z: Union[Sequence, np.ndarray] = []  # redshift sampling
    frequencies: Union[Sequence[int], np.ndarray] = np.array([],dtype=int)
    lmax: int = 0  # Maximum multipole value
    # extra_args: dict = {}  # extra (non-parameter) arguments passed to ccl.Cosmology()

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, "packages_path", None)):
            raise LoggedError(
                self.log,
                "No path given to Halo Model data. Set the likelihood property 'path' or the common property '%s'.",
                "packages_path",
            )

        # If no path specified, use the modules path
        self._data_file_path = os.path.normpath(
            getattr(self, "path", None)
            or os.path.join(self.packages_path, "data/halo_model")
        )
        if not os.path.exists(self._data_file_path):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self._data_file_path,
            )

        # Read input NFW
        '''
        with h5py.File(os.path.join(self._data_file_path, "halo_data_NFW.hdf5"), "r") as f:
            dataset = dict(_mh="mass_halo",_z="redshift",_k_unfw="k",_unfw="nfw_profile")
            for par,col in dataset.items():
                if col not in f:
                    raise LoggedError(self.log, f"Missing dataset '{col}' !")
                setattr(self, par, f[col][:])
                self.log.debug(f"{col} = {np.shape(getattr(self, par))}")
        #_mh: internal halo-mass range array(mh)
        #_z: internal z range for integrations array(nz)
        #_k_unfw: array(nk)
        #_unfw: array(m,k,z)
        #_snu_eff: array(nfq,nz)

        self._unfw = 0.*self._unfw
        '''

        #print("mh init", np.shape(self._mh), self._mh)

        #MD
        z1 = getattr(self, "zmin", 1e-5)
        z2 = getattr(self, "zmax", 5)
        nbpas_z = getattr(self, "nbpas_z", 102)
   
        powlogz = 0.25 #! 1.0
        AA=(z2/z1)**(1./(nbpas_z-1)**powlogz)
        redshift = np.zeros(nbpas_z)
        for jz in range(nbpas_z):
            redshift[jz]=z1*AA**(jz**powlogz) #!test
        self._z = redshift

        # M range for CIB
        logmass = np.arange(6, 13, 0.1) 
        mass = 10**logmass
        # M range for tSZ
        M1 = getattr(self, "Mmin_sz", 1e13)
        M2 = getattr(self, "Mmax_sz", 1e15/0.6733)
        nbpas_M = getattr(self, "nbpas_M", 100)
   
        logM1 = np.log10(M1)
        logM2 = np.log10(M2)
        masse = np.zeros(nbpas_M)
        for jm in range(nbpas_M):
            masse[jm]=10**(logM1+(jm)*(logM2-logM1)/(nbpas_M-1.))
            
        self._m  = np.hstack([mass,masse])


        
        

        #MD
        k_min = 1e-5; k_max = 100;
        nbpas_k = 200
        self._k = np.logspace(np.log(k_min), np.log(k_max), nbpas_k, base=np.e)

        
        self._ell = []


        #MD:
        self.juska = getattr(self, "juskaXr500", 5)

        xmax = np.log10(self.juska)
        self._x = np.logspace(-6, xmax, 50)
        #
        self._delta_h_cib = 200
        self._delta_h_tsz = 500

        self.tinker = Tinker08()
 
        # Scale quantities related to redshifts
       # self._m500c = np.tile(self._mh, (len(self._z), 1)).T

        # reading tabulated y_ell integration
        #self._yint_tab = np.loadtxt(os.path.dirname(__file__)+'/y_ell_integration.txt')

        # Defaults Halo parameters
        self._parameters2 = {
            "logMmax": 12.94217128427922,  # Meffmax=8753289339381.791
            "etamax": 0.4028353504978569,
            "sigmaMh": 1.807080723258688,
            "tauMh": 1.2040244128818796,
            #"B": 1.41,
            "B": 1.41,#1./0.982,
        }

        self.B = self._parameters2["B"]

        gamma = 0.31
        alpha = 1.33
        beta = 4.13
        P_0 = 6.41
        c_500 = 1.81
        pp_a = 1.0510
        pp_b = 5.4905
        self.coeffs_PP = [beta, alpha, P_0, alpha, beta, gamma, c_500]

        self.mode = []
        self.log.info("HaloModel loaded succesfully")
        
        self.r_sigma_t=6.6524   # !reduced Thomson cross-section
        self.r_me=9.11  # !reduced electron mas
        self.r_c_light=2.99792458 #!reduce speed of light
        self.G = 4.301e-9 #km2 Mpc MSun-1 s


    def get_requirements(self):
        """
        Get a dictionary of requirements that are always needed (e.g. must be calculated
        by a another component or provided as input parameters).

        :return: dictionary of requirements (or iterable of requirement names if no
                 optional parameters are needed)
        """
        # These are currently required to construct a new cosmology model.
        #return {"omegab", "omegam", "omegal", "H0", "mnu"}.union(self._parameters.keys())
        requirements = \
                        {'omegam': None, 
                         'H0': None,
                         'sigma8':None,
                         'ombh2': None,
                         'omch2': None,
                         'ns': None,
                         'omegab': None,
                         "omegam": None,
                         "omegal": None,
                         "mnu": None,
                         'angular_diameter_distance': {'z': self._z},
                         'Hubble': {'z': self._z, 'units': 'km/s/Mpc'},
                         }
        
        return requirements
 

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        options = requirements.get("Cl_from_halo_model") or {}

        newmodes = options.get("mode", [])
        for m in newmodes:
            if m not in ["CIB","tSZ","tSZxCIB"]:
                raise LoggedError(self.log, f"Mode should be CIB, tSZ or tSZxCIB ('{m}') !")
        self.mode = np.unique( np.concatenate( [newmodes,self.mode]))

        self.lmax = max(self.lmax, options.get("lmax", 13500))
        self._ell = np.unique( np.concatenate([options.get("ells",_default_ell_sampling), self._ell]))
        self._ell = self._ell[self._ell <= self.lmax]
        self.log.debug(f"lmax = {self.lmax}")

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))

        # Dictionary of the things HM needs from CAMB/CLASS
        self.nonlinear = self.nonlinear or options.get("nonlinear", False)
        needs = {}
        needs["Pk_interpolator"] = {
            "vars_pairs": [("delta_tot", "delta_tot")],
            "nonlinear": self.nonlinear,
            #"z": self._z[np.linspace(0,len(self._z)-1,120,dtype=int)],
            "z": self._z,
            "k_max": self.kmax,
        }

        lnRmin= 0.005
        lnRmax= 4
        dlnR= 0.01666667

        self.lnR_array = np.linspace(lnRmin,lnRmax,91)

        # needs['comoving_radial_distance'] = {'z': self.z}
        needs["Hubble"]={'z': self._z, 'units': 'km/s/Mpc'}
        
        #needs["sigma_R"]={'z': self._z[np.linspace(0,len(self._z)-1,120,dtype=int)],'k_max': self.kmax, 'R': np.exp(self.lnR_array),'vars_pairs': (['delta_tot', 'delta_tot'])}
        needs["sigma_R"]={'z': self._z,'k_max': self.kmax, 'R': np.exp(self.lnR_array),'vars_pairs': (['delta_tot', 'delta_tot'])}
        
 
        return needs

    def calculate(self, state, want_derived=True, **params_values_dict):
        t0 = time.time()


        
        self.log.debug("Calculating inside HM model")
        self._parameters = params_values_dict
        self.log.debug(f"parameters={self._parameters}")

        # CMB parameters are required and thus accessible via the params_values_dict
        Ob0 = self._parameters["omegab"]
        Om0 = self._parameters["omegam"]
        Ode0 = self._parameters["omegal"]
        H0 = self._parameters["H0"]  # Km.s-1.Mpc-1 ??
#        self.log.debug(f"omegab = {Ob0}")
#        self.log.debug(f"omegam = {Om0}")
#        self.log.debug(f"omegal = {Ode0}")
#        self.log.debug(f"H0 = {H0}")

        print("H0", H0)

        
        Om0_in = self.provider.get_param("omegam")
        h100_in = H0 /100.
        Hz_theory = self.provider.get_Hubble(self._z) # units [km/s/Mpc]
        Ez_theory = Hz_theory/(H0)
        rhocrit0 = 3 * H0**2 /8 /np.pi / self.G

        self._mh = self._m#/h100_in

        
        # Scale quantities related to redshifts
        self._m500c = np.tile(self._mh, (len(self._z), 1)).T

        
        self.mean_density0 = self.provider.get_param("omegam") * rhocrit0

        
        rhom0 = Om0_in*rhocrit0 # [Msun/Mpc^3]
        dA_theory = self.provider.get_angular_diameter_distance(self._z) # units [Mpc]
        dVdzdO_theory = self.r_c_light*1e8*1e-3/H0 * (1.+self._z)**2 * \
            dA_theory**2 / Ez_theory * 4 * np.pi # units [Mpc^3]  #MD *4pi for full sky

        

        state["chi"] = self.provider.get_angular_diameter_distance(self._z)*(1+self._z)

        # Get the matter power spectrum:
        t1 = time.time()
        Pk_interp = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"), 
            nonlinear=self.nonlinear, 
            extrap_kmin=np.min(self._k), 
            extrap_kmax=np.max(self._k)
        )
        # Get Pk at z, k
        state["Pk"] = Pk_interp.P(self._z, self._k) # inteprol 1
#        self.log.debug(f"Pk = {self._Pk}")
        if verbose_timing: print( "\tGet Pk ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        # Get Pk at z, kk=l/chi
        power = []
        for iz, z in enumerate(self._z):
            kk = self._ell / state["chi"][iz]
            #kk[kk>500]=499.
            #print("kk", kk) #kk[kk>500]=kk[400]
            #pk0 = Pk_interp.P(z, kk)
            pk0=interp1d(self._k, Pk_interp.P(self._z, self._k)[iz,:], axis=0, fill_value="extrapolate")
            pk000 = pk0(kk)
            #pk0[(self._ell / state["chi"][iz])>500]=0.
            power.append(pk000) # interpole 2 why ?
            
                
        state["power"] = np.array(power).T
        if verbose_timing: print( "\tGet Power ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        # Update bias and halo mass function
        self._update_hmf_bias( state)
        if verbose_timing: print( "\tCompute HMF and Bias ({:.4f}s)".format(time.time()-t1))




        #MD-------------------------------------------------
        #    calling above from var..
        # allow to loop on z

        self.u_nfw_cib = np.zeros((len(self._mh), len(self._k), len(self._z)))

        delta_h = 200 # put later to use self._delta_h_cib
    
        for r in range(len(self._z)):
            Pkk =  state["Pk"][r,:]
            self.u_nfw_cib[:, :, r] = self.nfwfourier_u(Pkk,self._z[r])
            

        if save_outputs: np.save("unfw.npy", self.u_nfw_cib)
        #MD
  

        t1 = time.time()
        # interpolate NFW at k = l/chi array(l,mh,z))
        state["unfw"] = np.zeros((len(self._ell), len(self._mh), len(self._z)))
        for iz in range(len(self._z)):
            kk = self._ell / state["chi"][iz]
            for im in range(len(self._mh)):
                #nfw_k = self._unfw[im, :, iz]
                nfw_md = self.u_nfw_cib[im, :, iz] 
                #state["unfw"][:, im, iz] = np.interp(kk, self._k_unfw, nfw_k)
                state["unfw"][:, im, iz] = np.interp(kk, self._k, nfw_md)
                
        if verbose_timing: print( "\tInterpolate NFW ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        if "tSZ" in self.mode or "tSZxCIB" in self.mode:

            state["y_ell"] = self._y_ell_tab(self._delta_h_tsz)  # array(ell,mh,z)
        if verbose_timing: print( "\tCompute y_ell ({:.4f}s)".format(time.time()-t1))

#        self.current_state = state

        if verbose_timing: print( "HaloModel ({:.4f}s)".format( time.time()-t0))


    def get_Cl_from_halo_model(self,instrument={}):
        """
        Get dictionary of Cl computed quantities.
        results['halo_model'] contains the halo model object object and results
        Other entries are computed by methods passed in as the requirements
        :return: dict of results

        #instrument should have the following arguments: nu, mode, fsz, Kcmb_MJy, cc, snu_eff
        """
        t0 = time.time()

        name = instrument["name"]
        aa=0.
        if aa==1.234 : #name in self.current_state:
            #MD cl = self.current_state[name]
            aaa=0
        else:
            try:
                cl = self.compute_cls( instrument)
            except:
                raise LoggedError(
                    self.log, "No Cl's were computed. Are you sure that you have requested them?"
                )
            self.current_state[name] = cl

        if verbose_timing: print( "\t \t Compute cls ({:.4f}s)".format(time.time()-t0))
        return cl

    def compute_cls(self, inst):
        '''
        Compute cross-spectra for CIB, tSZ and tSZxCIB

        Inputs:
            inst: dict{"name":str,
                       "mode":["CIB","tSZ","tSZxCIB"],
                       "nu":[],
                       "snu":dict{"nu","z","snu_eff"},
                       "cc":[],
                       "fsz":[],
                       "Kcmb_MJy":[]}

        Outputs:
            cl_cib: array(nfq,nfq,nell) [units K^2]
            cl_tsz: array(nfq,nfq,nell) [units K^2]
            cl_txc: array(nfq,nfq,nell) [units K^2]
        '''
        nu = inst["nu"]
        nfreq = len(nu)
        mode = inst["mode"]

        try:
            ifreq = [inst["snu"]["nu"].tolist().index(fq) for fq in nu]
        except:
            raise LoggedError( self.log, f"frequencies not found in SNU file: {nu}")
        #Interpolate SNU on internal redshift values
        snu_eff = np.array([InterpolatedUnivariateSpline(inst["snu"]["z"],inst["snu"]["snu_eff"][ifq])(self._z) for ifq in ifreq])

        #Instrument coef
        cc       = np.array(inst.get( "cc",       np.ones(nfreq)))
        Kcmb_MJy = np.array(inst.get( "Kcmb_MJy", np.ones(nfreq)))
        fsz      = np.array(inst.get( "fsz",      np.ones(nfreq)))
        ccunit   = cc/(Kcmb_MJy * 1e6)


        unfw = self.current_state["unfw"]
        chi  = self.current_state["chi"]

        H0_theory = self._parameters["H0"] 
        Hz_theory = self.provider.get_Hubble(self._z) # units [km/s/Mpc]
        Ez_theory = Hz_theory/(H0_theory)
        rho_crit_z = 3 *self.provider.get_Hubble(self._z)**2/8 /np.pi / self.G

        Cls = dict(ell=self._ell)

        t1 = time.time()
        if "CIB" in mode or "tSZxCIB" in mode:
            dj_cen = self._djc_dlnMh( snu_eff)    # array(freq,mh,z)
            dj_sub = self._djsub_dlnMh( snu_eff)  # array(freq,mh,z)
        if verbose_timing: print( "\tCompute dj ({:.4f}s)".format(time.time()-t1))
        t1 = time.time()
        if "CIB" in mode:
            Jv = self._J_nu(unfw, dj_cen, dj_sub)  # array(freq,l)
        if verbose_timing: print( "\tCompute Jnu ({:.4f}s)".format(time.time()-t1))


        Ez = Ez_theory
        dm = np.log10(self._mh[1] / self._mh[0])
        geo   = constants.c * 1e-3 / H0_theory / Ez / (chi * (1 + self._z)) ** 2  # array(z)
        dVcdz = constants.c * 1e-3 / H0_theory / Ez * chi** 2

        
        _power = self.current_state["power"]

        # CIB
        t1 = time.time()
        if "CIB" in mode:
            cl_cib = {}
            for f1, f2 in cwr(range(nfreq), 2):
                cl_one = np.zeros(len(self._ell))
                cl_two = np.zeros(len(self._ell))
                for il in range(len(self._ell)):
                    cib_intgmh = ( dj_cen[f1] * dj_sub[f2] * unfw[il, :, :] +
                                   dj_cen[f2] * dj_sub[f1] * unfw[il, :, :] +
                                   dj_sub[f1] * dj_sub[f2] * unfw[il, :, :]** 2
                                   ) * self.current_state["hmfmz_cib"]  # array(mh,z)
                    #cib_intgz1 = intg.simpson(cib_intgmh, dx=dm, axis=0) * geo
                    cib_intgz1 = intg.simpson(cib_intgmh, x=np.log10(self._mh), axis=0) * geo
                    
                    cib_intgz2 = Jv[f1, il, :] * Jv[f2, il, :] * geo * _power[il]

                    cl_one[il] = ccunit[f1] * intg.simpson(cib_intgz1, self._z) * ccunit[f2]   # one halo
                    cl_two[il] = ccunit[f1] * intg.simpson(cib_intgz2, self._z) * ccunit[f2]   # two halo
                cl_cib[(nu[f1],nu[f2])] = cl_one + cl_two
            if save_outputs: np.savetxt( f"cl_{inst['name']}_cib.dat", np.array([v for k,v in cl_cib.items()]))
            Cls.update( dict(CIB=cl_cib))
        if verbose_timing: print( "\tCompute cl(CIB) ({:.4f}s)".format(time.time()-t1))

        # tSZ
        t1 = time.time()
        if "tSZ" in mode:
            y_ell = self.current_state["y_ell"]
            cl_tsz = {}
            tsz_intgmh1 = self.current_state["hmfmz_tsz"][None,:,:] * y_ell ** 2
            tsz_intgmh2 = self.current_state["hmfmz_tsz"][None,:,:] * y_ell * self.current_state["biasmz_tsz"][None,:,:]
            #print("dm", np.shape(dm))
            tsz_intgz1 = intg.simpson(tsz_intgmh1, x=np.log10(self._mh), axis=1) * dVcdz
            tsz_intgz2 = intg.simpson(tsz_intgmh2, x=np.log10(self._mh), axis=1) ** 2 * dVcdz * _power
            cl_one = intg.simpson(tsz_intgz1, self._z)   # one halo
            cl_two = intg.simpson(tsz_intgz2, self._z) #MD   # two halo
            for f1, f2 in cwr(range(nfreq), 2):
                #print( fsz[f1] , fsz[f2])
                cl_tsz[(nu[f1],nu[f2])] = fsz[f1] * fsz[f2] * (cl_one + cl_two)
            if save_outputs: np.savetxt( f"cl_{inst['name']}_tsz.dat", np.array([v for k,v in cl_tsz.items()]))
            Cls.update( dict(tSZ=cl_tsz))
        if verbose_timing: print( "\tCompute cl(tSZ) ({:.4f}s)".format(time.time()-t1))

        # tSZxCIB
        t1 = time.time()
        if "tSZxCIB" in mode:
            y_ell = self.current_state["y_ell"]   # array(l,mh,z)
            hmfbias_cib = self.current_state["biasmz_cib"] * self.current_state["hmfmz_cib"]   # array(mh,z)
            hmfbias_tsz = self.current_state["biasmz_tsz"] * self.current_state["hmfmz_tsz"]   # array(mh,z)
            cl_txc = {}
            cosm = (1 + self._z) * chi**2  # array(z)

            xx = np.log10(self._mh)
            
            for f1, f2 in cwr(range(nfreq), 2): 
                djcensub = (dj_cen[f2][None,:,:] + dj_sub[f2][None,:,:] * unfw) * ccunit[f2] * fsz[f1] + \
                           (dj_cen[f1][None,:,:] + dj_sub[f1][None,:,:] * unfw) * ccunit[f1] * fsz[f2]   # array(l,mh,z)
                txc_intgmh1 = y_ell * djcensub / cosm * self.current_state["hmfmz_tsz"]  # 1h term hmf with deltah=500
                txc_intgmh2 = y_ell * hmfbias_tsz  # 2h term has hmf and bias with deltah=200 for cib (below) halo and 500 for tsz halo
                txc_intgmh3 = djcensub * hmfbias_cib / cosm

                txc_intgz1 = intg.simpson(txc_intgmh1, x=xx, axis=1) * dVcdz
                txc_intgz2 = ( intg.simpson(txc_intgmh2, x=xx, axis=1) * \
                               intg.simpson(txc_intgmh3, x=xx, axis=1) * \
                               dVcdz * _power)

                cl_one = intg.simpson(txc_intgz1, self._z)
                cl_two = intg.simpson(txc_intgz2, self._z)
                cl_txc[(nu[f1],nu[f2])] = cl_one + cl_two
            if save_outputs: np.savetxt( f"cl_{inst['name']}_txc.dat", np.array([v for k,v in cl_txc.items()]))
            Cls.update( dict(tSZxCIB=cl_txc))
        if verbose_timing: print( "\tCompute cl(tSZxCIB) ({:.4f}s)".format(time.time()-t1))

        return Cls

    def _update_hmf_bias(self, state):
        """
        Update bias and Halo Mass Function

        Output:
            _biasmz_cib: array(m,z)
            _hmfmz_cib: array(m,z)
        """


        Pk = state["Pk"]
        critical_density0 = 3 *self._parameters["H0"]**2 /8 /np.pi / self.G

        mean_density0 = self.provider.get_param("omegam") * critical_density0
        
        radius = (3 * self._mh / (4 * np.pi * mean_density0)) ** (1.0 / 3.0)

        def bias(delta_halo, sigma):
            y = np.log10(delta_halo)
            yy = np.exp(-((4.0 / y) ** 4))
            A = 1.0 + 0.24 * y * yy
            aa = 0.44 * y - 0.88
            C = 0.019 + 0.107 * y + 0.19 * yy
            # critical density of the universe. Redshift evolution is small and neglected
            dc = 1.686
            nuu = dc / sigma
            B = 0.183
            b = 1.5
            c = 2.4
            return 1 - (A * nuu ** aa / (nuu ** aa + dc ** aa)) + B * nuu ** b + C * nuu ** c

        Nmz = (len(self._mh), len(self._z))
        state["hmfmz_cib"] = np.zeros( Nmz)
        state["hmfmz_tsz"] = np.zeros( Nmz)
        state["biasmz_cib"] = np.zeros( Nmz)
        state["biasmz_tsz"] = np.zeros( Nmz)
        #sigma1, sigma2 = self._sigmas(radius, Pk)  # array(m,z)



        # computes sigma and dsigma/dM

        h100 = self._parameters["H0"]/100.
        _,_,sigma_nodes = self.provider.get_sigma_R()
                
        sigmaR_interpR = interp1d(np.exp(self.lnR_array),sigma_nodes,axis=1,fill_value="extrapolate")

        #print("nodes:", np.shape(sigma_nodes))

        sigmaR_arr = sigmaR_interpR(radius) # r in Mpc

        #print(np.shape(sigmaR_arr))

        dsigmadR_arr = np.gradient(sigmaR_arr, radius,axis=1, edge_order=1)
        
        dRdM = 1./3. * radius/self._mh
                
        dsigmadM_arr = dsigmadR_arr*dRdM

        #print(dsigmadM_arr)

        #sigmaC = np.zeros(Nmz)
        
        #sigmaC = interp1d(self._z[np.linspace(0,len(self._z)-1,120,dtype=int)], np.log(sigmaR_arr), axis=0, fill_value="extrapolate")
        sigmaC = interp1d(self._z, np.log(sigmaR_arr), axis=0, fill_value="extrapolate")

        #print("c", sigmaC)

        sigma1 = np.exp(sigmaC(self._z)).T

        #print("s1", sigma1)

        #dsigma = interp1d(self._z[np.linspace(0,len(self._z)-1,120,dtype=int)], np.log(-dsigmadM_arr), axis=0, fill_value="extrapolate")
        dsigma = interp1d(self._z, np.log(-dsigmadM_arr), axis=0, fill_value="extrapolate")
        
        sigma2 = (np.exp(dsigma(self._z)).T)/sigma1*self._mh[:,None]*3/0.5
        #print("s2",sigma2)
        
        for im, mass in enumerate(self._mh):
            # CIB
            Ez = self.provider.get_Hubble(self._z)/self.provider.get_param("H0")
            Om_z = np.array(self.provider.get_param("omegam")  *(1.+self._z)**3. / Ez**(2.))
            
            fsig = self.tinker.fsigma(self._z, sigma1[im], self._delta_h_cib / Om_z)
            
            dn_dm = fsig * mean_density0 * np.abs(0.5 * sigma2[im] / 3) / mass ** 2
            state["hmfmz_cib"][im] = mass * dn_dm * np.log(10)
            state["biasmz_cib"][im] = bias(self._delta_h_cib / Om_z, sigma1[im])

            # tSZ
            fsig = self.tinker.fsigma(self._z, sigma1[im], self._delta_h_tsz / Om_z)
            dn_dm = fsig * mean_density0 * np.abs(0.5 * sigma2[im] / 3) / mass ** 2
            state["hmfmz_tsz"][im] = mass * dn_dm * np.log(10)
            state["biasmz_tsz"][im] = bias(self._delta_h_tsz / Om_z, sigma1[im])
            # state["biasmz_tsz"][im] = state["biasmz_cib"][im]   #WHY ?




        #print("sigma")
        #print(sigma1.shape)
        #print(sigmaR_arr.shape)

        #np.save("hmf_tsz.npy", { "hmf": state["hmfmz_tsz"], "masse":self._mh, "redshift":self._z})

        #np.save("sigma2.npy", { "sigma": sigma2, "masse":self._mh, "redshift":self._z})
        #np.save("sigmaC.npy", { "sigma": sigmaR_arr.T, "masse":self._mh, "redshift":self._z[np.linspace(0,len(self._z)-1,120,dtype=int)], "dsigma":dsigmadM_arr.T})
        


        
        

    def _sigmas(self, radius, Pk):
        """
        matter variance for given power spectrum, wavenumbers k and radius of the halo.

        Inputs:
            radius: array(mh)
            Pk: array(nz,k)
        Outputs:
            sigma: array(mh,z)

        Integrate Pk with window function
        v1: log integ: Pk.k^3.W(rk).dlnk
        v2: lin integ: Pk.k^2.W(rk).dk (because k-vec not in log space)
        """
        self.log.debug("Computing sigmas...")
        rk = np.outer(radius, self._k)  # array(mh,k)

        # limit of 1.4e-6 put as it doesn't add much to the final answer and helps for faster convergence
        wrk1 = np.where(rk > 1.4e-6, (3 * (np.sin(rk) - rk * np.cos(rk)) / rk ** 3), 1)
        integ1 = Pk[None, :, :] * self._k** 2 * wrk1[:, None, :] ** 2

        wrk2 = np.where( rk > 1e-3, (9 * rk * np.cos(rk) + 3 * np.sin(rk) * (rk ** 2 - 3)) / rk ** 3, 0)
        integ2 = Pk[None, :, :] * self._k** 2 * wrk1[:, None, :] * wrk2[:, None, :]

#        sigma1 = np.sqrt((0.5 / np.pi ** 2) * intg.simpson(integ1, self._k))
#        sigma2 = 1.0 / (np.pi * sigma1) ** 2 * intg.simpson(integ2, self._k)
        #speed-up x1.5
        sigma1 = np.sqrt((0.5 / np.pi ** 2) * np.array([intg.simpson(itg, self._k) for itg in integ1]))
        sigma2 = 1.0 / (np.pi * sigma1) ** 2 * np.array([intg.simpson(itg, self._k) for itg in integ2])

        return sigma1, sigma2

    def _djsub_dlnMh(self, snu_eff):
        """
        for subhalos, the SFR is calculated in two ways and the minimum of the two is assumed.
        (not multiplied by hmf)
        """
        nfreq,nz = np.shape(snu_eff)

        dj_sub = np.zeros((nfreq, len(self._mh), nz))

        chi  = self.current_state["chi"]

        for i, mh in enumerate(self._mh):
            ms = self._msub(mh * (1 - self.fsub))  # array1d
            sfrI = self._sfr(ms)  # array(ms,z)
            sfrII = self._sfr(mh * (1 - self.fsub)) * ms[:, None] / (mh * (1 - self.fsub))
            subhmf = self._subhmf(mh, ms)
            sfrsub = np.minimum(sfrI, sfrII)  # array(ms,z)

            integral = sfrsub * subhmf[:, None] / self.KC
            #dlnmsub = np.log10(ms[1] / ms[0])
            #intgn = intg.simpson(integral, dx=dlnmsub, axis=0)
            intgn = intg.simpson(integral, x=np.log10(ms), axis=0)
            
            dj_sub[:, i, :] = snu_eff * (1+self._z) * intgn * chi**2

        return dj_sub

    def _djc_dlnMh(self, snu_eff):
        """
        Fraction of the mass of the halo that is in form of sub-halos. We have
        to take this into account while calculating the star formation rate of
        the central halos. It should be calulated by accounting for this
        fraction of the subhalo mass in the halo mass central halo mass in this
        case is (1-f_sub)*mh where mh is the total mass of the halo.
        for a given halo mass, f_sub is calculated by taking the first moment
        of the sub-halo mf and and integrating it over all the subhalo masses
        and dividing it by the total halo mass.
        (not multiplied by hmf)
        """

        chi  = self.current_state["chi"]

        mhalo = self._mh * (1 - self.fsub)

        rest = self._sfr(mhalo) * (1 + self._z) / self.KC
        dj_cen = rest * snu_eff[:, None, :] * chi**2
        return dj_cen

    def _sfr(self, mhalo):
        
        mhalo = np.atleast_1d(mhalo)
        sfrmhdot = self._sfr_mhdot(mhalo)
        mhdot = self._Mdot(mhalo)

        return mhdot * sfrmhdot * self._parameters["omegab"]/self._parameters["omegam"]*(self._z/self._z)
    
    def _sfr_mhdot(self, mhalo):
        """
        SFR/Mhdot lognormal distribution wrt halomass
        """

        Meffmax = 10 ** (self._parameters2["logMmax"])
        etamax = self._parameters2["etamax"]
        sigmaMh = self._parameters2["sigmaMh"]
        tauMh = self._parameters2["tauMh"]
        z_c = 1.5
        sigpow = sigmaMh - np.where(self._z < z_c, z_c - self._z, 0) * tauMh
        #sigpow = sigmaMh - np.array([max(0., z_c-z) for z in self._z_])*tauMh

        '''
        a = np.zeros((len(mhalo), len(self._z)))
        for i in range(len(mhalo)):
            if mhalo[i] < Meffmax:
                a[i, :] = etamax * np.exp(
                    -((np.log(mhalo[i]) - np.log(Meffmax)) ** 2) / (2 * sigmaMh ** 2)
                )
            else:
                a[i, :] = etamax * np.exp(
                    -((np.log(mhalo[i]) - np.log(Meffmax)) ** 2) / (2 * sigpow ** 2)
                )
        ''' 
        # Faster way (/2 time)
        a = etamax * np.exp(-((np.log(mhalo[:, None]) - np.log(Meffmax)) ** 2) / (2 * sigpow ** 2))
        a[mhalo < Meffmax] = etamax * np.exp(
             -((np.log(mhalo[mhalo < Meffmax, None]) - np.log(Meffmax)) ** 2) / (2 * sigmaMh ** 2)
         )

        return a

    def _Mdot(self, mhalo):
        # Use mean ?
        Om0 = self._parameters["omegam"]  
        Ode0 = self._parameters["omegal"] 

        a = 46.1 * (1 + 1.11 * self._z) * np.sqrt(Om0 * (1 + self._z) ** 3 + Ode0)
        b = (mhalo / 1e12) ** 1.1
        return np.outer(b, a)

    def _J_nu(self, unfw, dj_cen, dj_sub):
        """
        Inputs:
            unfw: array(l,m,z)
            dj_cen: array(f,m,z)
            dj_sub: array(f,m,z)
        Outputs:
            Jnu= array(freq,l,z)
        """
        nfreq, nm, nz = np.shape( dj_cen)
        Jnu = np.zeros((nfreq, len(self._ell), nz))

        dm = np.log10(self._mh[1] / self._mh[0])
        hmfbias = self.current_state["hmfmz_cib"] * self.current_state["biasmz_cib"]  # array(m,z)
        for il in range(len(self._ell)):
            rest1 = (dj_cen + dj_sub * unfw[il, :, :]) * hmfbias    # array(fq,m,z)
            #intg_mh = intg.simpson(rest1, dx=dm, axis=1)
            intg_mh = intg.simpson(rest1, x=np.log10(self._mh), axis=1)
            
            Jnu[:, il, :] = intg_mh

        return Jnu

    def _msub(self, mhalo):
        """
        for a given halo mass mh, the subhalo masses would range from
        m_min to mh. For now, m_min has been taken as 10^5 solar masses
        """
        log10msub_min = 5
        if np.any(np.log10(mhalo) <= log10msub_min):
            raise LoggedError(
                self.log,
                "halo mass {} should be greater than subhalo mass {}.".format(
                    np.log10(mhalo), log10msub_min
                ),
            )

        logmh = np.log10(mhalo)
        logmsub = np.arange(log10msub_min, logmh, 0.1)
        return 10 ** logmsub

    def _subhmf(self, mhalo, ms):
        # subhalo mass function from (https://arxiv.org/pdf/0909.1325.pdf)
        # np.log(10) added in the end as in the integration we are integrating with
        # respect to dlogm to the base 10.
        return 0.13 * (ms / mhalo) ** (-0.7) * np.exp(-9.9 * (ms / mhalo) ** (2.5)) * np.log(10)

    def _y_ell(self, delta_h):
        """
        Outputs:
            y_ell = array(l,m,z)
        """

        r500 = self._r_delta(delta_h)  # array(mh,nz)
        
        l500 = self.provider.get_angular_diameter_distance(self._z)/r500

        Mpc_to_m = 3.086e22  # Mpc to m
        sigT = constants.value("Thomson cross section")
        electron_mass = constants.electron_mass * constants.c ** 2
        a = (sigT / electron_mass) * 4 * np.pi * (r500 * Mpc_to_m) / l500 ** 2  # array(mh,nz)

        Pe = self._P_e()               # array(mh,nz,x)
        Pex2 = Pe * self._x ** 2
        x_ls = self._x / l500[:, :, None]  # array(mh,nz,x)

#        integral = [Pex2 * np.sin(ell * x_ls) / (ell * x_ls) for ell in self._ell]
#        y_ell = intg.simpson(integral, x=self._x) * a

        #quicker by a factor 2
        y_ell = np.array([intg.simpson(Pex2 * np.sin(ell * x_ls) / (ell * x_ls), x=self._x) * a for ell in self._ell])

        return y_ell

    def _C(self):
        h = self._parameters["H0"] / 100
   

        Ez = self.provider.get_Hubble(self._z)/self.provider.get_param("H0")
#        M_tilde = self._m500c / self._parameters["B"]  # array(m,z)
        M_tilde = self._m500c / self.B  # array(m,z)
        a = 1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3)
        b = (h / 0.7 * M_tilde / 3e14) ** (2.0 / 3 + 0.12)
        eV_to_J = 1.6e-19  # eV in  J
        cm_to_m = 0.01
        
        res = a * b * eV_to_J / cm_to_m ** 3  # converting to SI units
        return res  # dim m,z final units are SI # eV*cm**-3



    def yelldex(self): #MD
        
        #yy=np.linspace(-np.log(1e6),np.log(5e5),100)
        yy=np.linspace(-13.816,13.816,200) # 200
        yy=np.exp(yy)

        pp_beta, pp_alpha, pp_P0p, pp_a, pp_b, pp_g, pp_c500 = self.coeffs_PP
        xx1     = np.logspace(-6,np.log10(self.juska),10000)
        profile = (pp_c500 * xx1) ** (-pp_g) * (1.0 + (pp_c500 * xx1) ** pp_a) ** ((pp_g - pp_b) / pp_a)
        part11  = profile*xx1
        t0= time.time()
        intgn = np.zeros(np.size(yy))
        for i in range(len(yy)):
            #print(i)
            integral = part11*np.sin(yy[i] * xx1)/(yy[i])
            yyi=yy[i]
            intgn[i] = intg.simpson(integral, x=xx1) 
        ytilde=np.log(intgn)
        ytilde[yy>1e5]=-35.0
        ytilde[np.where(np.isnan(ytilde))[0][0]:]=-35.0
        if verbose_timing: print("\ttime integral yelldex", time.time()-t0)
        #print(np.size(np.array([yy,ytilde]).T))
        return np.array([np.log(yy),ytilde]).T

    
    def _y_ell_tab(self, delta_h ): # MD maybe r500*B
        r500 = self._r_delta(delta_h)  # array(mh,nz)
        
        l500 = self.provider.get_angular_diameter_distance(self._z)/r500

        Mpc_to_m = 3.086e22  # Mpc to m
        sigT = constants.value("Thomson cross section")
        electron_mass = constants.electron_mass * constants.c ** 2
        a = (sigT / electron_mass) * 4 * np.pi * (r500 * Mpc_to_m) / l500 ** 2  # array(mh,nz)

        C_t = self._C()
        #P_0 = 6.41
        #yl = self._yint_tab
        P_0 = self.coeffs_PP[2] * (self._parameters["H0"] / 70.)**(-3/2.)
        yl = self.yelldex()
        intgn = np.zeros((len(self._ell), len(self._mh), len(self._z)))

        for i in range(len(self._ell)):
            for j in range(len(self._mh)):
                l_l500 = self._ell[i]/l500[j, :]  # z
                y_int = np.interp(np.log(l_l500), yl[:, 0], yl[:, 1])
                intgn[i, j, :] = np.exp(y_int)
        return P_0*intgn*a*C_t  # dim ell,m,z  unitless

    def _r_delta(self, delta_h):
        """
        radius of the halo containing amount of matter corresponding to delta times
        the critical density of the universe inside that halo
        (miss delta_h)

        Output:
            r_delta: array(mh,z) [Units: Mpc]
        """
#        M_tilde = self._m500c / self._parameters["B"]
        M_tilde = self._m500c / self.B
        critical_density = 3 *self.provider.get_Hubble(self._z)**2 /8 /np.pi / self.G

        r3 = 3 * M_tilde / (4 * np.pi * delta_h * critical_density)
        
               
        return r3 ** (1 / 3)

    def _P_e(self):
        """
        Output:
            P_e: array(mh,z,x) [Unit:J/m^3]
        """

   
        Ez = self.provider.get_Hubble(self.z_array)/self._parameters["H0"]

        # constants from https://www.aanda.org/articles/aa/pdf/2013/02/aa20040-12.pdf
        #gamma_t = 0.31
        #alpha_t = 1.33
        #beta_t = 4.13
        #P_0_t = 6.41
        #c_500_t = 1.81

        h = self._parameters["H0"] / 100

        eV_to_J = 1.6e-19  # eV in  J
        cm_to_m = 0.01

#        M_tilde = self._m500c / self._parameters["B"]  # array(m,z)
        M_tilde = self._m500c / self.B  # array(m,z)
        a1 = 1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3)
        b1 = (h / 0.7 * M_tilde / 3e14) ** (2.0 / 3 + 0.12)
        C_t = a1 * b1 * eV_to_J / cm_to_m ** 3  # converting to SI units

        a = C_t[:, :, None] * P_0_t / (c_500_t * self._x) ** (gamma_t)
        b = (1 + (c_500_t * self._x) ** alpha_t) ** ((gamma_t - beta_t) / alpha_t)

        return a * b



    # ################ NFW profile calculation #######################

    """
    Code to calculate the Fourier transform of the NFW profile. The analytical
    formula has been taken from arXiv:1206.6890v1 (or arXiv:astro-ph/0006319v2)
    From https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz/blob/master/hmf_unfw_bias.py
    """
    def mass_to_radius(self):
        """
        Lagrangian radius of a dark matter halo
        """
        rho_mean = self.mean_density0
        r3 = 3*self._mh/(4*np.pi*rho_mean)
        return r3**(1./3.)

    def sine_cosine_int(self, x):
        r"""
        sine and cosine integrals required to calculate the Fourier transform
        of the NFW profile.
        $ si(x) = \int_0^x \frac{\sin(t)}{t} dt \\
        ci(x) = - \int_x^\infty \frac{\cos(t)}{t}dt$
        """
        si, ci = sici(x)
        return si, ci

    def k_R(self):
        """
        we need a c-M relation to calculate the fourier transform of the NFW
        profile. We use the relation from https://arxiv.org/pdf/1407.4730.pdf
        where they use the slope of the power spectrum with respect to the
        wavenumber in their formalism. This power spectrum slope has to be
        evaluated at a certain value of k such that k = kappa*2*pi/rad
        This kappa value comes out to be 0.69 according to their calculations.
        """
        rad = self.mass_to_radius()
        kappa = 0.69
        return kappa * 2 * np.pi / rad

    def dlnpk_dlnk(self,Pkk): #MDshould be done for each z here Pk is for one z
        """
        When the power spectrum is obtained from CAMB, slope of the ps wrt k
        shows wiggles at lower k which corresponds to the BAO features. Also
        at high k, there's a small dip in the slope of the ps which is due to
        the effect of the baryons (this is not very important for the current
        calculations though). We are using the analysis from the paper
        https://arxiv.org/pdf/1407.4730.pdf where they have used the power
        spectrum from Eisenstein and Hu 1998 formalism where these effects have
        been negelected and therefore they don't have the wiggles and the bump.
        In order to acheive this, we have to smooth out the
        slope of the ps at lower k. But we have checked that the results
        do not vary significantly with the bump.
        """
        grad = np.zeros(Pkk.shape, np.float)
        grad[0:-1] = np.diff(np.log(Pkk)) / np.diff(np.log(self._k))
        grad[-1] = (np.log(Pkk[-1]) - np.log(Pkk[-2]))/(np.log(self._k[-1]) - np.log(self._k[-2]))
        # grad_smoothed = savgol_filter(grad, 51, 2)
        grad_smoothed = grad
        kr = self.k_R()
        
        return np.interp(kr, self._k, grad_smoothed)


    def r_delta(self, delta_h, z):
        """
        radius of the halo containing amount of matter corresponding to delta times
        the critical density of the universe inside that halo
        """
        
        rho_crit_z = 3 *self.provider.get_Hubble(z)**2/8 /np.pi / self.G

        
        r3 = 3*self._mh/(4*np.pi*delta_h*rho_crit_z)
        return r3**(1./3.)


    

    def nu_delta(self,Pkk,z):  # peak heights

        rad = self.r_delta(200,z)
        """
        to calculate the peak heights: nu_delta,  we use r_delta rather than
        the simple Lagrangian radius calculated using mass_to_radius function.
        This will be used in c-M relation to calculate the NFW profile.
        """

        rk = np.outer(rad, self._k)
        rest = Pkk * self._k**3
        # dlnk = np.log(self.kk[1] / self.kk[0])
        lnk = np.log(self._k)
        Wrk = (3 * (np.sin(rk) - rk * np.cos(rk)) / rk ** 3)
        integ = rest*Wrk**2
        sig = (0.5/np.pi**2) * intg.simpson(integ, x=lnk, axis=-1)
        
        delta_c = 1.686  # critical density of the universe. Redshift evolution
        # is small and neglected

        return delta_c / sig  # length of mass array

    
    def nu_to_c200c(self,Pkk,z):  # length of mass array
        use_mean = False  # 2 relations provided. mean and median.
        phi0_median, phi0_mean = 6.58, 7.14
        phi1_median, phi1_mean = 1.37, 1.60
        eta0_median, eta0_mean = 6.82, 4.10
        eta1_median, eta1_mean = 1.42, 0.75
        alpha_median, alpha_mean = 1.12, 1.40
        beta_median, beta_mean = 1.69, 0.67
        _nu = self.nu_delta(Pkk, z)
        n_k = self.dlnpk_dlnk(Pkk)
        #print(np.shape(n_k), np.shape(_nu))
        if use_mean:
            c_min = phi0_mean + phi1_mean * n_k
            nu_min = eta0_mean + eta1_mean * n_k
            return c_min * \
                ((_nu/nu_min)**-alpha_mean + (_nu/nu_min)**beta_mean)/2
        else:
            c_min = phi0_median + phi1_median * n_k
            nu_min = eta0_median + eta1_median * n_k
            return c_min * \
                ((_nu/nu_min)**-alpha_median + (_nu/nu_min)**beta_median)/2

        #print("nu2c200 OK")
    def r_star(self,Pkk, z):
        r"""
        characteristic radius also calld r_s in other literature.
        $ c \equiv \frac{r_{200}{r_s}$
        """
        c_200c = self.nu_to_c200c(Pkk, z)
        r200 = self.r_delta(200,z)
        return r200/c_200c  # length of mass array

    def ampl_nfw(self, c):
        r"""
        Dimensionless amplitude of the NFW profile.
        Gives:
            $\frac{1}{\log(1+c) - \frac{c}{1+c}}$
        """
        return 1. / (np.log(1.+c) - c/(1.+c))

    def nfwfourier_u(self,Pkk, z):
        rs = self.r_star(Pkk,z)
        c = self.nu_to_c200c(Pkk,z)
        a = self.ampl_nfw(c)
        mu = np.outer(self._k, rs)

        Si1, Ci1 = self.sine_cosine_int(mu + mu * c)
        Si2, Ci2 = self.sine_cosine_int(mu)
        unfw = a*(cos(mu)*(Ci1-Ci2) + sin(mu)*(Si1-Si2)-sin(mu*c) / (mu+mu*c))
      
        return unfw.transpose()  # dim(len(m), len(k))

    


