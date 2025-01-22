import os
import time
from itertools import combinations_with_replacement as cwr
from typing import Sequence, Union

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import LambdaCDM
from cobaya.log import LoggedError
from cobaya.theory import Theory
from scipy import constants
from scipy import integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline

verbose_timing = True
save_outputs = False

# fmt: off
#k values to which get the Pk for integration inside the halo-model
_default_k_sampling = np.array([
    7.04746845e-06, 7.55289345e-06, 8.09456614e-06, 8.67508610e-06, 9.29723935e-06, 9.96401172e-06, 1.06786032e-05, 1.14444431e-05,
    1.22652070e-05, 1.31448339e-05, 1.40875451e-05, 1.50978650e-05, 1.61806422e-05, 1.73410732e-05, 1.85847272e-05, 1.99175726e-05,
    2.13460060e-05, 2.28768826e-05, 2.45175496e-05, 2.62758806e-05, 2.81603143e-05, 3.01798944e-05, 3.23443132e-05, 3.46639581e-05,
    3.71499616e-05, 3.98142544e-05, 4.26696229e-05, 4.57297706e-05, 4.90093836e-05, 5.25242015e-05, 5.62910923e-05, 6.03281343e-05,
    6.46547017e-05, 6.92915586e-05, 7.42609581e-05, 7.95867492e-05, 8.52944913e-05, 9.14115770e-05, 9.79673632e-05, 1.04993312e-04,
    1.12523143e-04, 1.20592993e-04, 1.29241590e-04, 1.38510440e-04, 1.48444027e-04, 1.59090023e-04, 1.70499521e-04, 1.82727276e-04,
    1.95831972e-04, 2.09876501e-04, 2.24928265e-04, 2.41059500e-04, 2.58347623e-04, 2.76875602e-04, 2.96732356e-04, 3.18013183e-04,
    3.40820211e-04, 3.65262897e-04, 3.91458544e-04, 4.19532871e-04, 4.49620612e-04, 4.81866162e-04, 5.16424275e-04, 5.53460799e-04,
    5.93153482e-04, 6.35692814e-04, 6.81282949e-04, 7.30142683e-04, 7.82506503e-04, 8.38625711e-04, 8.98769635e-04, 9.63226914e-04,
    1.03230689e-03, 1.10634110e-03, 1.18568483e-03, 1.27071887e-03, 1.36185133e-03, 1.45951955e-03, 1.56419227e-03, 1.67637182e-03,
    1.79659659e-03, 1.92544354e-03, 2.06353104e-03, 2.21152179e-03, 2.37012604e-03, 2.54010494e-03, 2.72227426e-03, 2.91750826e-03,
    3.12674391e-03, 3.35098536e-03, 3.59130879e-03, 3.84886755e-03, 4.12489772e-03, 4.42072401e-03, 4.73776614e-03, 5.07754566e-03,
    5.44169324e-03, 5.83195647e-03, 6.25020830e-03, 6.69845601e-03, 7.17885080e-03, 7.69369818e-03, 8.24546900e-03, 8.83681129e-03,
    9.47056303e-03, 1.01497657e-02, 1.08776789e-02, 1.16577960e-02, 1.24938610e-02, 1.33898863e-02, 1.43501720e-02, 1.53793267e-02,
    1.64822896e-02, 1.76643540e-02, 1.89311928e-02, 2.02888857e-02, 2.17439486e-02, 2.33033647e-02, 2.49746177e-02, 2.67657284e-02,
    2.86852926e-02, 3.07425227e-02, 3.29472917e-02, 3.53101806e-02, 3.78425293e-02, 4.05564912e-02, 4.34650909e-02, 4.65822873e-02,
    4.99230405e-02, 5.35033833e-02, 5.73404984e-02, 6.14528008e-02, 6.58600262e-02, 7.05833256e-02, 7.56453671e-02, 8.10704442e-02,
    8.68845929e-02, 9.31157163e-02, 9.97937187e-02, 1.06950649e-01, 1.14620855e-01, 1.22841146e-01, 1.31650974e-01, 1.41092619e-01,
    1.51211393e-01, 1.62055857e-01, 1.73678056e-01, 1.86133767e-01, 1.99482768e-01, 2.13789122e-01, 2.29121488e-01, 2.45553449e-01,
    2.63163865e-01, 2.82037252e-01, 3.02264186e-01, 3.23941740e-01, 3.47173948e-01, 3.72072306e-01, 3.98756306e-01, 4.27354008e-01,
    4.58002659e-01, 4.90849347e-01, 5.26051708e-01, 5.63778686e-01, 6.04211339e-01, 6.47543709e-01, 6.93983759e-01, 7.43754360e-01,
    7.97094371e-01, 8.54259782e-01, 9.15524937e-01, 9.81183861e-01, 1.05155166e+00, 1.12696605e+00, 1.20778894e+00, 1.29440824e+00,
    1.38723962e+00, 1.48672862e+00, 1.59335270e+00, 1.70762356e+00, 1.83008962e+00, 1.96133860e+00, 2.10200039e+00, 2.25275006e+00,
    2.41431108e+00, 2.58745882e+00, 2.77302423e+00, 2.97189788e+00, 3.18503420e+00, 3.41345607e+00, 3.65825972e+00, 3.92062002e+00,
    4.20179608e+00, 4.50313730e+00, 4.82608989e+00, 5.17220375e+00, 5.54313994e+00, 5.94067865e+00, 6.36672774e+00, 6.82333190e+00,
    7.31268244e+00, 7.83712786e+00, 8.39918506e+00, 9.00155145e+00, 9.64711789e+00, 1.03389826e+01, 1.10804659e+01, 1.18751264e+01,
    1.27267777e+01, 1.36395071e+01, 1.46176949e+01, 1.56660357e+01, 1.67895605e+01, 1.79936615e+01, 1.92841173e+01, 2.06671210e+01,
    2.21493099e+01, 2.37377974e+01, 2.54402067e+01, 2.72647082e+01, 2.92200579e+01, 3.13156398e+01, 3.35615112e+01, 3.59684502e+01,
    3.85480082e+01, 4.13125651e+01, 4.42753883e+01, 4.74506971e+01, 5.08537302e+01, 5.45008195e+01, 5.84094680e+01, 6.25984339e+01,
    6.70878210e+01, 7.18991745e+01, 7.70555850e+01, 8.25817990e+01, 8.85043379e+01, 9.48516250e+01, 1.01654122e+02, 1.08944475e+02,
    1.16757673e+02, 1.25131211e+02, 1.34105276e+02, 1.43722937e+02, 1.54030349e+02, 1.65076981e+02, 1.76915847e+02, 1.89603764e+02,
    2.03201623e+02, 2.17774683e+02, 2.33392883e+02, 2.50131176e+02, 2.68069895e+02, 2.87295128e+02, 3.07899142e+02, 3.29980820e+02,
    3.53646134e+02, 3.79008660e+02, 4.06190115e+02, 4.35320950e+02, 4.66540968e+02, 5.00000000e+02])
#    kmax = 500
#    logmink = np.log10(7.04746845e-06)
#    logkmax = np.log10(kmax)
#    diff = logkmax - logmink
#    kfin = np.logspace(logmink, np.log10(kmax), int(diff/step+1))
# fmt: on

# fmt: off
_default_ell_sampling = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 114, 187, 200, 320, 502, 684, 890, 1158, 1505, 1956, 3000, 5000, 10000])
# fmt: on

# fmt: off
_default_z_sampling = np.array([
    0.012,  0.035,  0.059,  0.084,  0.109,  0.135,  0.161,  0.189,
    0.216,  0.245,  0.274,  0.303,  0.334,  0.365,  0.396,  0.429,
    0.462,  0.496,  0.531,  0.567,  0.603,  0.641,  0.679,  0.718,
    0.758,  0.799,  0.841,  0.884,  0.928,  0.972,  1.018,  1.065,
    1.113,  1.163,  1.213,  1.265,  1.317,  1.371,  1.427,  1.483,
    1.541,  1.6  ,  1.661,  1.723,  1.786,  1.851,  1.917,  1.985,
    2.055,  2.126,  2.199,  2.273,  2.35 ,  2.428,  2.508,  2.589,
    2.673,  2.758,  2.846,  2.936,  3.027,  3.121,  3.217,  3.315,
    3.416,  3.519,  3.624,  3.732,  3.842,  3.955,  4.07 ,  4.188,
    4.309,  4.433,  4.559,  4.689,  4.821,  4.957,  5.095,  5.237,
    5.383,  5.531,  5.683,  5.839,  5.998,  6.161,  6.328,  6.499,
    6.674,  6.852,  7.035,  7.222,  7.414,  7.61 ,  7.81 ,  8.016,
    8.226,  8.441,  8.661,  8.886,  9.116,  9.351,  9.593,  9.839,
    10.092])
# fmt: on


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

        self._k = _default_k_sampling
        self._ell = []

        # Halo mass sampling (corresponding to NFW pre-computed files)
#        self._mh = 10 ** (np.arange(60, 151) / 10)
        self._x = np.logspace(-6, 1, 50)
        self._delta_h_cib = 200
        self._delta_h_tsz = 500

        self.tinker = Tinker08()

        # Scale quantities related to redshifts
        self._m500c = np.tile(self._mh, (len(self._z), 1)).T

        # reading tabulated y_ell integration
        self._yint_tab = np.loadtxt(os.path.dirname(__file__)+'/y_ell_integration.txt')

        # Defaults Halo parameters
        self._parameters = {
            "logMmax": 12.94217128427922,  # Meffmax=8753289339381.791
            "etamax": 0.4028353504978569,
            "sigmaMh": 1.807080723258688,
            "tauMh": 1.2040244128818796,
            "B": 1.41,
        }

        self.mode = []
        self.log.info("HaloModel loaded succesfully")


    def get_requirements(self):
        """
        Get a dictionary of requirements that are always needed (e.g. must be calculated
        by a another component or provided as input parameters).

        :return: dictionary of requirements (or iterable of requirement names if no
                 optional parameters are needed)
        """
        # These are currently required to construct a new cosmology model.
        return {"omegab", "omegam", "omegal", "H0"}.union(self._parameters.keys())

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

        self.lmax = max(self.lmax, options.get("lmax", 10000))
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
            "z": self._z[np.linspace(0,len(self._z)-1,120,dtype=int)],
            "k_max": self.kmax,
        }
        # needs['comoving_radial_distance'] = {'z': self.z}

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

        lcdm = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0, Ob0=Ob0) #WARNING approx LCDM (enough for halo_model)
        state["lcdm"] = lcdm
        state["chi"] = lcdm.comoving_distance(self._z).value

        # Get the matter power spectrum:
        Pk_interp = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"), nonlinear=self.nonlinear, extrap_kmin=np.min(self._k), extrap_kmax=np.max(self._k)
            )

        t1 = time.time()
        # Get Pk at z, k
        state["Pk"] = Pk_interp.P(self._z, self._k)
#        self.log.debug(f"Pk = {self._Pk}")
        if verbose_timing: print( "\tGet Pk ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        # Get Pk at z, kk=l/chi
        power = []
        for iz, z in enumerate(self._z):
            kk = self._ell / state["chi"][iz]
            power.append(Pk_interp.P(z, kk))
        state["power"] = np.array(power).T
        if verbose_timing: print( "\tGet Power ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        # Update bias and halo mass function
        self._update_hmf_bias( state)
        if verbose_timing: print( "\tCompute HMF and Bias ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        # interpolate NFW at k = l/chi array(l,mh,z))
        state["unfw"] = np.zeros((len(self._ell), len(self._mh), len(self._z)))
        for iz in range(len(self._z)):
            kk = self._ell / state["chi"][iz]
            for im in range(len(self._mh)):
                nfw_k = self._unfw[im, :, iz]
                state["unfw"][:, im, iz] = np.interp(kk, self._k_unfw, nfw_k)
        if verbose_timing: print( "\tInterpolate NFW ({:.4f}s)".format(time.time()-t1))

        t1 = time.time()
        if "tSZ" in self.mode or "tSZxCIB" in self.mode:
#            state["y_ell"] = self._y_ell(self._delta_h_tsz,lcdm)  # array(ell,mh,z)
            # Using the tabulated version significantly speeds up the calculation
            state["y_ell"] = self._y_ell_tab(self._delta_h_tsz,lcdm)  # array(ell,mh,z)
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
        if name in self.current_state:
            cl = self.current_state[name]

        else:
            try:
                cl = self.compute_cls( instrument)
            except:
                raise LoggedError(
                    self.log, "No Cl's were computed. Are you sure that you have requested them?"
                )
            self.current_state[name] = cl

        if verbose_timing: print( "Compute cls ({:.4f}s)".format(time.time()-t0))
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

        lcdm = self.current_state["lcdm"]
        unfw = self.current_state["unfw"]
        chi  = self.current_state["chi"]

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

        Ez = np.sqrt(lcdm.Om0 * (1 + self._z) ** 3 + lcdm.Ode0)
        dm = np.log10(self._mh[1] / self._mh[0])
        geo = constants.c * 1e-3 / lcdm.H0 / Ez / (chi * (1 + self._z)) ** 2  # array(z)
        dVcdz = constants.c * 1e-3 / lcdm.H0 / Ez * chi** 2

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
                    cib_intgz1 = intg.simps(cib_intgmh, dx=dm, axis=0) * geo
                    cib_intgz2 = Jv[f1, il, :] * Jv[f2, il, :] * geo * _power[il]

                    cl_one[il] = ccunit[f1] * intg.simps(cib_intgz1, self._z) * ccunit[f2]   # one halo
                    cl_two[il] = ccunit[f1] * intg.simps(cib_intgz2, self._z) * ccunit[f2]   # two halo
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
            tsz_intgz1 = intg.simps(tsz_intgmh1, dx=dm, axis=1) * dVcdz
            tsz_intgz2 = intg.simps(tsz_intgmh2, dx=dm, axis=1) ** 2 * dVcdz * _power
            for f1, f2 in cwr(range(nfreq), 2):
                cl_one = fsz[f1] * fsz[f2] * intg.simps(tsz_intgz1, self._z)   # one halo
                cl_two = fsz[f1] * fsz[f2] * intg.simps(tsz_intgz2, self._z)   # two halo
                cl_tsz[(nu[f1],nu[f2])] = cl_one + cl_two
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
            for f1, f2 in cwr(range(nfreq), 2):
                djcensub = (dj_cen[f2][None,:,:] + dj_sub[f2][None,:,:] * unfw) * ccunit[f2] * fsz[f1] + \
                           (dj_cen[f1][None,:,:] + dj_sub[f1][None,:,:] * unfw) * ccunit[f1] * fsz[f2]   # array(l,mh,z)
                txc_intgmh1 = y_ell * djcensub / cosm * self.current_state["hmfmz_tsz"]  # 1h term hmf with deltah=500
                txc_intgmh2 = y_ell * hmfbias_tsz  # 2h term has hmf and bias with deltah=200 for cib (below) halo and 500 for tsz halo
                txc_intgmh3 = djcensub * hmfbias_cib / cosm

                txc_intgz1 = intg.simps(txc_intgmh1, dx=dm, axis=1) * dVcdz
                txc_intgz2 = ( intg.simps(txc_intgmh2, dx=dm, axis=1) * \
                               intg.simps(txc_intgmh3, dx=dm, axis=1) * \
                               dVcdz * _power)

                cl_one = intg.simps(txc_intgz1, self._z)
                cl_two = intg.simps(txc_intgz2, self._z)
##                 cl_one = np.zeros( len(self._ell))
##                 cl_two = np.zeros( len(self._ell))
##                 for il in range(len(self._ell)):
##                     djcensub = (dj_cen[f2] + dj_sub[f2] * unfw[il, :, :]) * ccunit[f2] * fsz[f1] + \
##                                (dj_cen[f1] + dj_sub[f1] * unfw[il, :, :]) * ccunit[f1] * fsz[f2]
##                     txc_intgmh1 = y_ell[il] * djcensub / cosm * self.current_state["hmfmz_cib"]
##                     txc_intgmh2 = y_ell[il] * hmfbias
##                     txc_intgmh3 = djcensub / cosm * hmfbias

##                     txc_intgz1 = intg.simps(txc_intgmh1, dx=dm, axis=0) * dVcdz
##                     txc_intgz2 = ( intg.simps(txc_intgmh2, dx=dm, axis=0) * \
##                                    intg.simps(txc_intgmh3, dx=dm, axis=0) * \
##                                    dVcdz * _power[il])

##                     cl_one[il] = intg.simps(txc_intgz1, self._z)
##                     cl_two[il] = intg.simps(txc_intgz2, self._z)
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

        lcdm = state["lcdm"]
        Pk = state["Pk"]
        mean_density0 = lcdm.Om0 * lcdm.critical_density0.to(u.Msun / u.Mpc ** 3).value
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
        sigma1, sigma2 = self._sigmas(radius, Pk)  # array(m,z)
        for im, mass in enumerate(self._mh):
            # CIB
            fsig = self.tinker.fsigma(self._z, sigma1[im], self._delta_h_cib / lcdm.Om(self._z))
            dn_dm = fsig * mean_density0 * np.abs(0.5 * sigma2[im] / 3) / mass ** 2
            state["hmfmz_cib"][im] = mass * dn_dm * np.log(10)
            state["biasmz_cib"][im] = bias(self._delta_h_cib / lcdm.Om(self._z), sigma1[im])

            # tSZ
            fsig = self.tinker.fsigma(self._z, sigma1[im], self._delta_h_tsz / lcdm.Om(self._z))
            dn_dm = fsig * mean_density0 * np.abs(0.5 * sigma2[im] / 3) / mass ** 2
            state["hmfmz_tsz"][im] = mass * dn_dm * np.log(10)
            state["biasmz_tsz"][im] = bias(self._delta_h_tsz / lcdm.Om(self._z), sigma1[im])
            # state["biasmz_tsz"][im] = state["biasmz_cib"][im]   #WHY ?


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

#        sigma1 = np.sqrt((0.5 / np.pi ** 2) * intg.simps(integ1, self._k))
#        sigma2 = 1.0 / (np.pi * sigma1) ** 2 * intg.simps(integ2, self._k)
        #speed-up x1.5
        sigma1 = np.sqrt((0.5 / np.pi ** 2) * np.array([intg.simps(itg, self._k) for itg in integ1]))
        sigma2 = 1.0 / (np.pi * sigma1) ** 2 * np.array([intg.simps(itg, self._k) for itg in integ2])

        return sigma1, sigma2

    def _djsub_dlnMh(self, snu_eff):
        """
        for subhalos, the SFR is calculated in two ways and the minimum of the two is assumed.
        (not multiplied by hmf)
        """
        nfreq,nz = np.shape(snu_eff)

        dj_sub = np.zeros((nfreq, len(self._mh), nz))
        lcdm = self.current_state["lcdm"]
        chi  = self.current_state["chi"]

        for i, mh in enumerate(self._mh):
            ms = self._msub(mh * (1 - self.fsub))  # array1d
            sfrI = self._sfr(ms)  # array(ms,z)
            sfrII = self._sfr(mh * (1 - self.fsub)) * ms[:, None] / (mh * (1 - self.fsub))
            subhmf = self._subhmf(mh, ms)
            sfrsub = np.minimum(sfrI, sfrII)  # array(ms,z)

            integral = sfrsub * subhmf[:, None] / self.KC
            dlnmsub = np.log10(ms[1] / ms[0])
            intgn = intg.simps(integral, dx=dlnmsub, axis=0)
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
        lcdm = self.current_state["lcdm"]
        chi  = self.current_state["chi"]

        mhalo = self._mh * (1 - self.fsub)

        rest = self._sfr(mhalo) * (1 + self._z) / self.KC
        dj_cen = rest * snu_eff[:, None, :] * chi**2
        return dj_cen

    def _sfr(self, mhalo):
        lcdm = self.current_state["lcdm"]
        mhalo = np.atleast_1d(mhalo)
        sfrmhdot = self._sfr_mhdot(mhalo)
        mhdot = self._Mdot(mhalo)
        return mhdot * sfrmhdot * lcdm.Ob(self._z) / lcdm.Om(self._z)

    def _sfr_mhdot(self, mhalo):
        """
        SFR/Mhdot lognormal distribution wrt halomass
        """

        Meffmax = 10 ** (self._parameters["logMmax"])
        etamax = self._parameters["etamax"]
        sigmaMh = self._parameters["sigmaMh"]
        tauMh = self._parameters["tauMh"]
        z_c = 1.5
        sigpow = sigmaMh - np.where(self._z < z_c, z_c - self._z, 0) * tauMh
        #sigpow = sigmaMh - np.array([max(0., z_c-z) for z in self._z_])*tauMh

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

        # Faster way (/2 time)
        # a = etamax * np.exp(-((np.log(mhalo[:, None]) - np.log(Meffmax)) ** 2) / (2 * sigpow ** 2))
        # a[mhalo < Meffmax] = etamax * np.exp(
        #     -((np.log(mhalo[mhalo < Meffmax, None]) - np.log(Meffmax)) ** 2) / (2 * sigmaMh ** 2)
        # )

        return a

    def _Mdot(self, mhalo):
        # Use mean ?
        Om0 = self._parameters["omegam"]  # self.lcdm.Om0
        Ode0 = self._parameters["omegal"]  # self.lcdm.Ode0

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
            intg_mh = intg.simps(rest1, dx=dm, axis=1)
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

    def _y_ell(self, delta_h, lcdm):
        """
        Outputs:
            y_ell = array(l,m,z)
        """

        r500 = self._r_delta(delta_h,lcdm)  # array(mh,nz)
        l500 = lcdm.angular_diameter_distance(self._z).value / r500  # array(mh,nz)

        Mpc_to_m = 3.086e22  # Mpc to m
        sigT = constants.value("Thomson cross section")
        electron_mass = constants.electron_mass * constants.c ** 2
        a = (sigT / electron_mass) * 4 * np.pi * (r500 * Mpc_to_m) / l500 ** 2  # array(mh,nz)

        Pe = self._P_e(lcdm)               # array(mh,nz,x)
        Pex2 = Pe * self._x ** 2
        x_ls = self._x / l500[:, :, None]  # array(mh,nz,x)

#        integral = [Pex2 * np.sin(ell * x_ls) / (ell * x_ls) for ell in self._ell]
#        y_ell = intg.simps(integral, x=self._x) * a

        #quicker by a factor 2
        y_ell = np.array([intg.simps(Pex2 * np.sin(ell * x_ls) / (ell * x_ls), x=self._x) * a for ell in self._ell])

        return y_ell

    def _C(self, lcdm):
        h = self._parameters["H0"] / 100
        Ez = np.sqrt(lcdm.Om0 * (1 + self._z) ** 3 + lcdm.Ode0)
        M_tilde = self._m500c / self._parameters["B"]  # array(m,z)
        a = 1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3)
        b = (h / 0.7 * M_tilde / 3e14) ** (2.0 / 3 + 0.12)
        eV_to_J = 1.6e-19  # eV in  J
        cm_to_m = 0.01
        
        res = a * b * eV_to_J / cm_to_m ** 3  # converting to SI units
        return res  # dim m,z final units are SI # eV*cm**-3

    def _y_ell_tab(self, delta_h, lcdm):
        r500 = self._r_delta(delta_h,lcdm)  # array(mh,nz)
        l500 = lcdm.angular_diameter_distance(self._z).value / r500  # array(mh,nz)
        Mpc_to_m = 3.086e22  # Mpc to m
        sigT = constants.value("Thomson cross section")
        electron_mass = constants.electron_mass * constants.c ** 2
        a = (sigT / electron_mass) * 4 * np.pi * (r500 * Mpc_to_m) / l500 ** 2  # array(mh,nz)

        C_t = self._C(lcdm)
        P_0 = 6.41
        yl = self._yint_tab
        intgn = np.zeros((len(self._ell), len(self._mh), len(self._z)))

        for i in range(len(self._ell)):
            for j in range(len(self._mh)):
                l_l500 = self._ell[i]/l500[j, :]  # z
                y_int = np.interp(np.log(l_l500), yl[:, 0], yl[:, 1])
                intgn[i, j, :] = np.exp(y_int)
        return P_0*intgn*a*C_t  # dim ell,m,z  unitless

    def _r_delta(self, delta_h, lcdm):
        """
        radius of the halo containing amount of matter corresponding to delta times
        the critical density of the universe inside that halo
        (miss delta_h)

        Output:
            r_delta: array(mh,z) [Units: Mpc]
        """
        M_tilde = self._m500c / self._parameters["B"]
        r3 = 3 * M_tilde / (4 * np.pi * delta_h * lcdm.critical_density(self._z).to(u.Msun / u.Mpc ** 3).value)
        return r3 ** (1 / 3)

    def _P_e(self, lcdm):
        """
        Output:
            P_e: array(mh,z,x) [Unit:J/m^3]
        """

        Ez = np.sqrt(lcdm.Om0 * (1 + self._z) ** 3 + lcdm.Ode0)

        # constants from https://www.aanda.org/articles/aa/pdf/2013/02/aa20040-12.pdf
        gamma_t = 0.31
        alpha_t = 1.33
        beta_t = 4.13
        P_0_t = 6.41
        c_500_t = 1.81

        h = self._parameters["H0"] / 100

        eV_to_J = 1.6e-19  # eV in  J
        cm_to_m = 0.01

        M_tilde = self._m500c / self._parameters["B"]  # array(m,z)
        a1 = 1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3)
        b1 = (h / 0.7 * M_tilde / 3e14) ** (2.0 / 3 + 0.12)
        C_t = a1 * b1 * eV_to_J / cm_to_m ** 3  # converting to SI units

        a = C_t[:, :, None] * P_0_t / (c_500_t * self._x) ** (gamma_t)
        b = (1 + (c_500_t * self._x) ** alpha_t) ** ((gamma_t - beta_t) / alpha_t)

        return a * b
