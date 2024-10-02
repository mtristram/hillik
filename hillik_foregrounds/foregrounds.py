# FOREGROUNDS CMB LIK
import astropy.io.fits as fits
from astropy import constants
import os
import numpy as np
import itertools
import warnings
from cobaya.log import HasLogger, LoggedError


t_cmb = 2.72548
k_b = constants.k_B.value  # J / K
h_pl = constants.h.si.value  # J s


# equivalent frequencies (experiment dependant)
fsz = {
    "PLK": {100: 100.2, 143: 143, 217: 222},
    "SPT": {95: 96.55, 150: 152.26, 220: 220.1}, # col5 in spt_hiell_2020.info (dusty_clustered, dusty_poisson, radio, ksz, tsz)
    "SPT3G": {90: 96.481, 150: 148.949, 220: 219.578}, # SPT3G_2018_TTTEEE_effective_band_centres.dat
    "ACTw": {98: 98.4, 150: 149.9},
    "ACTd": {98: 98.4, 150: 150.1},
    }

fdust = {
    "PLK": {100: 105.2, 143: 148.5, 217: 228.1, 353: 370.5}, #alpha=4 from [Planck 2013 IX]
    "SPT": {95: 96.89, 150: 153.37, 220: 221.6, 353: 353}, #col1 in spt_hiell_2020.info (dusty_clustered, dusty_poisson, radio, ksz, tsz)
    "SPT3G": {90: 96.67, 150: 149.9, 220: 222.0, 353: 353}, #SPT3G_2018_TTTEEE_effective_band_centres.dat
    "ACTw": {98: 98.6, 150: 150.8, 353: 353},
    "ACTd": {98: 98.6, 150: 151.1, 353: 353},
    }

fradio = {
    "PLK": {100: 100.4, 143: 140.5, 217: 218.6},
    "SPT": {95: 93.5, 150: 149.46, 220: 215.8}, #col3 in spt_hiell_2020.info (dusty_clustered, dusty_poisson, radio, ksz, tsz)
    "SPT3G": {90: 94.4, 150: 146.0, 220: 212.7}, #SPT3G_2018_TTTEEE_effective_band_centres.dat
    "ACTw": {98: 95.8, 150: 147.2},
    "ACTd": {98: 95.8, 150: 147.1},
    }

fcib = fdust
fsync = fradio

#Parameter names
#{survey}_A{fg}_{freq(xfreq)}

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(HasLogger):
    """
    Class of foreground model for the Hillik likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    #frequency reference for fg amplitudes
    # ACT: 150 dust, 150 tSZ
    # SPT: 220 dust, 143 tSZ
    # SPT3G: 150 dust, 143 tSZ
#    feff = 150
    feff = 143

    def _f_tsz(self, freq):
        # Freq in GHz
        nu = freq*1e9
        xx = h_pl*nu/(k_b*t_cmb)
        return xx*( 1/np.tanh(xx/2.) ) - 4

    def _f_Planck(self, f, T):
        # Freq in GHz
        nu = f*1e9
        xx = h_pl*nu / (k_b*T)
        return (nu**3.)/(np.exp(xx)-1.)

    #Temp Antenna conversion
    def _dBdT(self, f):
        # Freq in GHz
        nu = f*1e9
        xx = h_pl*nu / (k_b*t_cmb)
        return (nu)**4 * np.exp(xx) / (np.exp(xx)-1.)**2.

    def _tszRatio(self, f, f0):
        return self._f_tsz(f)/self._f_tsz(f0)

    def _cibRatio(self, f, f0, beta=1.75, T=25): #T=9.7 ?!?
        return (f/f0)**beta * (self._f_Planck(f, T)/self._f_Planck(f0, T))\
            / (self._dBdT(f)/self._dBdT(f0))
    
    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f, T)/self._f_Planck(f0, T))\
            / ( self._dBdT(f)/self._dBdT(f0) )
    
    def _radioRatio(self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def _syncRatio(self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def __init__(
            self, lmax, freqs, mode="TT", auto=False, survey="",
            filename=None, lnorm=3000, **kwargs
        ):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.name = None
        self.survey = survey
        self.lnorm = lnorm

        # effecftive frequencies
        if self.survey not in fdust.keys():
            raise ValueError( f"Missing DUST effective frequency for {self.survey}")
        if self.survey not in fsz.keys():
            raise ValueError( f"Missing SZ effective frequency for {self.survey}")
        if self.survey not in fradio.keys():
            raise ValueError( f"Missing RADIO effective frequency for {self.survey}")
        if self.survey not in fcib.keys():
            raise ValueError( f"Missing CIB effective frequency for {self.survey}")
        if self.survey not in fsync.keys():
            raise ValueError( f"Missing SYNC effective frequency for {self.survey}")
        self.fdust  = fdust[survey]
        self.fsz    = fsz[survey]
        self.fradio = fradio[survey]
        self.fcib   = fcib[survey]
        self.fsync  = fsync[survey]

        # Build the list of cross frequencies
        if survey == "PLK":
            # Remove auto-spectra and use separate TE and ET (e.g. Planck)
            self._cross_frequencies = list(itertools.combinations(freqs, 2))
            self.combine_TE_and_ET = False
        elif "ACT" in survey:
            # Use auto spectra with separated TE and ET (ACT)
            if mode == "TE":
                self._cross_frequencies = list(itertools.product(freqs, repeat=2))
            else:
                self._cross_frequencies = list(itertools.combinations_with_replacement(freqs, 2))
            self.combine_TE_and_ET = False
        elif "SPT" in survey:
            #Use auto spectra with combined TE and ET (SPT)
            self._cross_frequencies = list(itertools.combinations_with_replacement(freqs, 2))
            self.combine_TE_and_ET = True
        else:
            raise ValueError( f"Survey {survey} not supported")

        self.set_logger()
        pass

    def _gen_dl_powerlaw(self, alpha, lnorm=3000):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = self.lmax if lnorm is None else max(self.lmax, lnorm)
        ell = np.arange(2, lmax+1)

        template = np.zeros(lmax+1)
        template[np.array(ell, int)] = ell*(ell+1)/2/np.pi * ell**(alpha)

        # normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename, lnorm=3000):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """

        # read dl template
        l, data = np.loadtxt(filename, unpack=True)
        self.log.debug("Template: {}".format(filename))

        template = np.zeros(max(self.lmax, int(max(l))) + 1)
        template[np.array(l, int)] = data

        # normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass


# ------------------------------------------------------------------------------------------------




# Point Sources (for TT)
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "PS"
        self.dltemp = self._gen_dl_powerlaw(0., lnorm=lnorm)

    def compute_dl(self, pars):
        if self.mode == "TT":
            dl_ps = []
            for f1, f2 in self._cross_frequencies:
                dl_ps.append( pars[f"{self.survey}_ps_{f1}x{f2}"] * self.dltemp)
            return np.array(dl_ps)
        else:
            return 0.


# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "PS radio"
        self.ll2pi = self._gen_dl_powerlaw(0.,lnorm=lnorm)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._radioRatio(self.fradio[f1], self.feff, beta=pars['beta_radio'])
                * self._radioRatio(self.fradio[f2], self.feff, beta=pars['beta_radio'])
            )

        if self.mode == "TT":
            return pars[f"{self.survey}_radio_ps"] * np.array(dl)
        else:
            return 0.


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "PS dusty"
        self.ll2pi = self._gen_dl_powerlaw(0.,lnorm=lnorm)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._cibRatio(self.fcib[f1], self.feff, beta=pars['beta_dusty'])
                * self._cibRatio(self.fcib[f2], self.feff, beta=pars['beta_dusty'])
            )

        if self.mode == "TT":
            return pars[f"{self.survey}_cib_ps"] * np.array(dl)
        else:
            return 0.



#Galactic Dust model
#Dl ACT: -0.6 for TT, -0.4 for TE, -0.4 for EE
#Dl SPT: -1.2 for TT (cirrus but very low amplitude) ?!?
#Dl SPT3G: -0.53 for TT, -0.42 for TE, -0.42 for EE (fit with strong prior)
#Dl Planck: -0.63 for TT, -0.4 for TE
class dust(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=80):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "Dust"

        if filename is None:
            alpha_dust = -2.5 if mode == "TT" else -2.4
            self.dlg = self._gen_dl_powerlaw(alpha_dust,lnorm=lnorm)
        else:
            self.dlg = self._read_dl_template(filename)
        
    def compute_dl(self, pars):
        if   self.mode == "TT": beta1, beta2 = pars[f"{self.survey}_beta_dustT"], pars[f"{self.survey}_beta_dustT"]
        elif self.mode == "TE": beta1, beta2 = pars[f"{self.survey}_beta_dustT"], pars[f"{self.survey}_beta_dustP"]
        elif self.mode == "ET": beta1, beta2 = pars[f"{self.survey}_beta_dustP"], pars[f"{self.survey}_beta_dustT"]
        elif self.mode == "EE": beta1, beta2 = pars[f"{self.survey}_beta_dustP"], pars[f"{self.survey}_beta_dustP"]

        if   self.mode == "TT": ad1, ad2 = pars[f'{self.survey}_AdustT'], pars[f'{self.survey}_AdustT']
        elif self.mode == "TE": ad1, ad2 = pars[f'{self.survey}_AdustT'], pars[f'{self.survey}_AdustP']
        elif self.mode == "ET": ad1, ad2 = pars[f'{self.survey}_AdustP'], pars[f'{self.survey}_AdustT']
        elif self.mode == "EE": ad1, ad2 = pars[f'{self.survey}_AdustP'], pars[f'{self.survey}_AdustP']

        #PLK amplitude of Dl(l=10) at 353GHz
        PLK_dl353 = {'TT': {100: 108125, 143: 31700, 217: 14700},
                     'EE': {100: 995, 143: 610, 217: 380},
                     'TE': {100: 2400, 143: 1540, 217: 1000},
                     'ET': {100: 2400, 143: 1540, 217: 1000}}
        PLK_alpha = {'TT': {100: -2.6, 143: -2.4, 217: -2.3},
                     'EE': {f: -2.4 for f in [100, 143, 217]},
                     'TE': {f: -2.4 for f in [100, 143, 217]},
                     'ET': {f: -2.4 for f in [100, 143, 217]}}

        dl = []
        for xf, (f1, f2) in enumerate(self._cross_frequencies):
            #rescale PLK for each combination of mask
            if self.survey == "PLK":
                dlg = self._gen_dl_powerlaw( PLK_alpha[self.mode][max(f1, f2)], lnorm=self.lnorm)
                ad = PLK_dl353[self.mode][max(f1, f2)]/dlg[10]
            else:
                ad = 1.
                dlg = self.dlg

            dl.append(ad * ad1 * ad2 * dlg
                       * self._dustRatio(self.fdust[f1], self.fdust[353], beta=beta1)
                       * self._dustRatio(self.fdust[f2], self.fdust[353], beta=beta2)
                       )
        return np.array(dl)


#Dust amplitudes
class dust_amplitude(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=200):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "Dust Amplitudes"        
        self.dlg = np.zeros(lmax+1)

        ell = np.arange(2, lmax+1)
        alpha_dust = -2.5 if mode == "TT" else -2.4
        self.dlg = self._gen_dl_powerlaw(alpha_dust,lnorm=lnorm)
    
    def compute_dl(self, pars):
        if   self.mode == "TT": ad1, ad2 = f'{self.survey}_dustT', f'{self.survey}_dustT'
        elif self.mode == "TE": ad1, ad2 = f'{self.survey}_dustT', f'{self.survey}_dustP'
        elif self.mode == "ET": ad1, ad2 = f'{self.survey}_dustP', f'{self.survey}_dustT'
        elif self.mode == "EE": ad1, ad2 = f'{self.survey}_dustP', f'{self.survey}_dustP'

        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(pars[ad1+f"_{f1}"] * pars[ad2+f"_{f2}"] * self.dlg)

        return np.array(dl)


# Synchrotron model
class sync(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=200):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "Synchrotron"

        #check effective freqs
        for f in freqs:
            if f not in self.fsyn:
                raise ValueError(f"Missing SYNC effective frequency for {f}")

        alpha_syn = -2.5  # Cl template power-law TBC
        self.dl_syn = self._gen_dl_powerlaw( alpha_syn, lnorm=100)
        self.beta_syn = -0.7

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(self.dl_syn
                       * self._syncRatio(self.fsyn[f1], self.feff, beta=self.beta_syn)
                       * self._syncRatio(self.fsyn[f2], self.feff, beta=self.beta_syn)
                       )
        if self.mode == "TT":
            return pars[f"{self.survey}_AsyncT"] * np.array(dl)
        elif self.mode == "EE":
            return pars[f"{self.survey}_AsyncP"] * np.array(dl)
        else:
            return 0.


# CIB clustered (one spectrum for all freqs)
class cib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "clustered CIB"

        #check effective freqs
        for f in freqs:
            if f not in self.fcib.keys():
                raise ValueError( f"Missing CIB effective frequency for {f}")

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw( alpha_cib, lnorm=lnorm)
        else:
            self.dl_cib = self._read_dl_template( filename, lnorm=lnorm)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio(self.fcib[f1], self.feff, pars['beta_cib'], pars.get('T_cib',25.))
                       * self._cibRatio(self.fcib[f2], self.feff, pars['beta_cib'], pars.get('T_cib',25.))
                       )
        if self.mode == "TT":
            return pars["Acib"] * np.array(dl)
        else:
            return 0.


#thermal SZ (one spectrum for all freqs)
class tsz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "tSZ"

        #check effective freqs
        for f in freqs:
            if f not in self.fsz.keys():
                raise ValueError(f"Missing SZ effective frequency for {f}")

        sz_tmpl = self._read_dl_template(filename, lnorm=lnorm)

        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append(
                sz_tmpl
                * self._tszRatio(self.fsz[f1], self.feff)
                * self._tszRatio(self.fsz[f2], self.feff)
            )
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            derived = {"Atsz_derived": 0.}
            return pars["Atsz"] * self.dl_sz, derived
        else:
            return 0., {}


#emulated thermal SZ (one spectrum for all freqs)
class tsz_emulator(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "tSZ"

        self.lmax = int(lmax)
        if self.lnorm is None:
            self.lnorm = 3000
        if filename is None:
            self.tsz_emulator = None
        else:
            self.tsz_emulator = init_emulator(filename, verbose=bool(self.log.level))

    def compute_dl(self, pars):

        if self.mode == "TT":
            self.dl_tsz = []
            derived = {}
            if self.tsz_emulator is None:
                self.dl_tsz = np.array([np.zeros(self.lmax+1) for f1, f2 in self._cross_frequencies])
            else:
                ref_tsz = self.tsz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=np.arange(self.lmax+1),
                    with_unit=True,
                    T_cmb=t_cmb,
                )  # to uK2
                for f1, f2 in self._cross_frequencies:
                    self.dl_tsz.append(ref_tsz *
                        self._f_tsz(self.fsz[f1]) * self._f_tsz(self.fsz[f2])
                    )
                D3000 = self.tsz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=[self.lnorm],
                    with_unit=True,
                    T_cmb=t_cmb,
                )[0] * self._f_tsz(self.feff)**2  # uK2
                derived = {"Atsz_derived": D3000}
            return self.dl_tsz, derived
        else:
            return 0., {}


#(homogeneous) kinetic SZ (one spectrum for all freqs)
class ksz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "kSZ"

        self.dl_ksz = []
        ksz_tmpl = self._read_dl_template( filename, lnorm=lnorm)
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksz_tmpl)
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            prefactor = pars["Aksz"]
            derived = {"Ahksz_derived": 0.}
            return prefactor * self.dl_ksz, derived
        else:
            return 0., {}


# emulated homogeneous kinetic SZ (one spectrum for all freqs)
class hksz_emulator(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "hkSZ"

        self.lmax = int(lmax)
        if self.lnorm is None:
            self.lnorm = 3000
        if filename is None:
            self.ksz_emulator = None
        else:
            self.ksz_emulator = init_emulator(filename, verbose=bool(self.log.level))

    def compute_dl(self, pars):
        if self.mode == "TT":
            self.dl_ksz = []
            derived = {}
            if self.ksz_emulator is None:
                self.dl_ksz = np.array([np.zeros(self.lmax+1) for f1, f2 in self._cross_frequencies])
            else:
                self.dl_ksz = np.array([self.ksz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=np.arange(self.lmax+1),
                    with_unit=True,
                    T_cmb=t_cmb,
                ) for f1, f2 in self._cross_frequencies])
                D3000 = self.ksz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=[self.lnorm],
                    with_unit=True,
                    T_cmb=t_cmb,
                )[0]  # uK2
                derived.update({"Ahksz_derived": D3000})
            return self.dl_ksz, derived
        else:
            return 0., {}


#patchy kinetic SZ (one spectrum for all freqs)
class pksz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "pkSZ"

        self.dl_ksz = []
        self.lmax = int(lmax)
        if filename is None:
            ksz_tmpl = np.zeros(lmax+1)
        else:
            ksz_tmpl = self._read_dl_template(filename, lnorm=lnorm)
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksz_tmpl)
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            derived = {"Apksz_derived": 0.}
            return pars["Apksz"] * self.dl_ksz, derived
        else:
            return 0., {}


#emulated patchy kinetic SZ (one spectrum for all freqs)
class pksz_emulator(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "pkSZ"

        self.lmax = int(lmax)
        if self.lnorm is None: 
            self.lnorm = 3000
        if filename is None:
            self.ksz_emulator = None
        else:
            self.ksz_emulator = init_emulator(filename, verbose=bool(self.log.level))

    def compute_dl(self, pars):
        if self.mode == "TT":
            self.dl_ksz = []
            derived = {}
            if self.ksz_emulator is None:
                self.dl_ksz = np.array([np.zeros(self.lmax+1) for f1, f2 in self._cross_frequencies])
            else:
                self.dl_ksz = np.array([self.ksz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=np.arange(self.lmax+1),
                    with_unit=True,
                    T_cmb=t_cmb,
                ) for f1, f2 in self._cross_frequencies])
                D3000 = self.ksz_emulator.get_cls(
                    cosmo_dict=pars,
                    ells=[self.lnorm],
                    with_unit=True,
                    T_cmb=t_cmb,
                )[0]  # uK2
                derived.update({"Apksz_derived": D3000})
            return self.dl_ksz, derived
        else:
            return 0., {}


# SZxCIB model (one spectrum for all freqs)
class szxcib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000, **kwargs):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "SZxCIB"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz.keys():
                raise ValueError( f"Missing SZ effective frequency for {f}")

        #check effective freqs for cib
        for f in freqs:
            if f not in self.fcib.keys():
                raise ValueError( f"Missing Dust effective frequency for {f}")

        self._is_template = bool(filename)
        if self._is_template:
            self.x_tmpl = self._read_dl_template(filename, lnorm=lnorm)
            self.tsz_emulator = None
        elif "filenames" in kwargs:
            if kwargs['emulator']:
                self.tsz_emulator = init_emulator(kwargs["filenames"][0], verbose=bool(self.log.level))  # tsz emulator
                self.x_tmpl = self._read_dl_template(kwargs["filenames"][1], lnorm=lnorm)  # cib only
            else:
                self.tsz_emulator = None
                self.x_tmpl = self._read_dl_template(kwargs["filenames"][0], lnorm=lnorm)*self._read_dl_template(kwargs["filenames"][1], lnorm=lnorm)
        else:
            raise ValueError(f"Missing template for SZxCIB  for {self.survey}")
            
    def compute_dl(self, pars):
        dl_szxcib = []
        if self.tsz_emulator is not None:
            ref_tsz = self.tsz_emulator.get_cls(
                cosmo_dict=pars,
                ells=np.arange(self.lmax+1),
                with_unit=False,
            ) * 1e12  # uK2
            ref_tsz = self.x_tmpl * np.sqrt(ref_tsz)
            for u, (f1, f2) in enumerate(self._cross_frequencies):
                dl_szxcib.append( ref_tsz * np.sqrt(pars["Acib"]) * (
                    self._f_tsz(self.fsz[f2]) * self._cibRatio(self.fcib[f1], self.feff, pars['beta_cib']) +
                    self._f_tsz(self.fsz[f1]) * self._cibRatio(self.fcib[f2], self.feff, pars['beta_cib'])
                    ))
        else:
            ref_tsz = self.x_tmpl * np.sqrt(pars['Atsz'])
            for u, (f1, f2) in enumerate(self._cross_frequencies):
                dl_szxcib.append(ref_tsz * np.sqrt(pars["Acib"]) * (
                    self._tszRatio(self.fsz[f2],self.feff) * self._cibRatio(self.fcib[f1], self.feff, pars['beta_cib']) +
                    self._tszRatio(self.fsz[f1],self.feff) * self._cibRatio(self.fcib[f2], self.feff, pars['beta_cib'])
                    ))
        if self.mode == "TT":
            return -1. * pars["xi"] * np.array(dl_szxcib)
        else:
            return 0.


def init_emulator(filename, verbose=False):
    """
    Initialise emul_sz object from filename.

    Parameters
    ----------
        filename: str
            Absolute path to + rootname of emulator coefficient files
            (see emul_sz documentation.)
        verbose: boolean
            Verbosity level.
    Returns
    -------
        emul_sz object.
    """
    try:
        import emul_sz
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "If you want to use the emulator, you must install emul_sz."
        ) from e

    emul_folder = os.path.dirname(filename)
    if len(emul_folder) == 0:
        emul_folder = os.environ['COBAYA_DIR']+'modules/data/foregrounds/'
    return emul_sz.emulator(
        seed=os.path.basename(filename),
        folder=emul_folder,
        verbose=verbose,
    )