# FOREGROUNDS ACT DR4
import astropy.io.fits as fits
import numpy as np
import itertools
from cobaya.log import HasLogger, LoggedError

t_cmb = 2.725
k_b = 1.3806503e-23
h_pl = 6.626068e-34
t_eff = 9.7

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(HasLogger):
    """
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    # ACT values
    freqs = [98,150]
    feff = 150

    # equivalent frequencies
    fsz   = [98.4, 150.1]
    fdust = [98.8, 151.2]
    fsyn  = [95.8, 147.2]

    #spectral indices
    beta_g = 1.5
#    beta_p = beta_c
    alpha_s = -0.5
    alpha_gs = -1.0

    def _sz_func( self, freq):
        """
        Freq in GHz
        """

        nu = freq*1e9
        xx=h_pl*nu/(k_b*t_cmb)
        return xx*(1/np.tanh(xx/2.)) - 4

    #t_eff = 9.7
    #t_eff = 19.6
    def _PlanckFunctionRatio( self, freq,t_eff,fe=150):
        nu  = freq*1e9
        nu0 = fe*1e9
        xx  = h_pl*nu /(k_b*t_eff)
        xx0 = h_pl*nu0/(k_b*t_eff)

        return (nu/nu0)**3.*(np.exp(xx0)-1.)/(np.exp(xx)-1.)

    def _Flux2TempRatio(self, freq,fe=150):
        # rescaled to 150 GHz
        nu  = freq*1e9
        nu0 = fe*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        xx0 = h_pl*nu0/(k_b*t_cmb)
        return (nu0/nu)**4.*(np.exp(xx0)/np.exp(xx))*((np.exp(xx)-1.)/(np.exp(xx0)-1.))**2.
        

    def __init__(self, lmax, freqs, mode="TT", auto=True, survey=""):
        """
        Create model for foreground
        """
        self.mode   = mode
        self.lmax   = lmax
        self.freqs  = freqs
        self.survey = survey
        self.name   = None

        # Build the list of cross frequencies
        nfreq = len(freqs)
        if auto:
            if mode == "TE":
                self._cross_frequencies = list(itertools.product(range(nfreq), repeat=2))
            else:
                self._cross_frequencies = list(itertools.combinations_with_replacement(range(nfreq), 2))
        else:
            if mode == "TE":
                self._cross_frequencies = list(itertools.permutations(range(nfreq), 2))
            else:
                self._cross_frequencies = list(itertools.combinations(range(nfreq), 2))

        self.set_logger()
        pass

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass


# ------------------------------------------------------------------------------------------------


#CIB clustered (TT)
class CIBclustered(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "clustered_cib"

        ell = np.arange( lmax+1)
#        self.clc=(ell*(ell+1)/3000/3001)*(ell/3000)**(-1.2)
        self.clc = np.zeros( lmax+1)
        self.clc[2:] = np.loadtxt( filename, unpack=True)[1,:lmax-1]

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_c"] * self.clc *
                       (self.fdust[f1]*self.fdust[f2]/self.feff**2)**pars["beta_c"] *
                       self._PlanckFunctionRatio(self.fdust[f1],9.7) * self._PlanckFunctionRatio(self.fdust[f2],9.7) *
                       self._Flux2TempRatio( self.fdust[f1]) * self._Flux2TempRatio( self.fdust[f2])
                       )
        return np.array(dl)


#CIB poisson (TT)
class CIBpoisson(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "poisson_cib"

        ell = np.arange( lmax+1)
        self.clp=(ell*(ell+1)/3000/3001)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_d"] * self.clp *
                       (self.fdust[f1]*self.fdust[f2]/self.feff**2)**pars["beta_c"] *
                       self._PlanckFunctionRatio(self.fdust[f1],9.7) * self._PlanckFunctionRatio(self.fdust[f2],9.7) *
                       self._Flux2TempRatio( self.fdust[f1]) * self._Flux2TempRatio( self.fdust[f2])
                       )
        return np.array(dl)


#RADIO poisson (TT, TE, EE) wide/deep
class RADIOpoisson(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True, survey="wide"):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "poisson_radio"
        ell = np.arange( lmax+1)
        self.clp=(ell*(ell+1)/3000/3001)

    def compute_dl(self, pars):
        Aps = 0
        if self.survey == "deep":
            if self.mode == "TT": Aps = pars['a_s'] 
            if self.mode == "TE": Aps = pars['a_tps']
            if self.mode == "EE": Aps = pars['a_ps']
        if self.survey == "wide":
            if self.mode == "TT": Aps = pars['a_sw'] 
            if self.mode == "TE": Aps = pars['a_tps']
            if self.mode == "EE": Aps = pars['a_ps']
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( Aps * self.clp *
                       (self.fsyn[f1]*self.fsyn[f2]/self.feff**2)**self.alpha_s *
                       self._Flux2TempRatio( self.fsyn[f1]) * self._Flux2TempRatio( self.fsyn[f2])
                       )
        return np.array(dl)

#Galactic Dust (TT,TE,EE) wide/deep
class GalacticDust(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True, survey="deep"):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "galactic dust"

        ell = np.arange( 2, lmax+1)
        self.clg = np.zeros( lmax+1)
        if mode == "TT":
            self.clg[2:] = (ell/500)**(-0.6) #clgt
        else:
            self.clg[2:] = (ell/500)**(-0.4) #clgp
        
    def compute_dl(self, pars):
        Adust = 0
        if self.survey == "deep":
            if self.mode == "TT": Adust = pars["a_gd"]
            if self.mode == "TE": Adust = pars["a_gted"]
            if self.mode == "EE": Adust = pars["a_geed"]
        if self.survey == "wide":
            if self.mode == "TT": Adust = pars["a_gw"]
            if self.mode == "TE": Adust = pars["a_gtew"]
            if self.mode == "EE": Adust = pars["a_geew"]
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( Adust * self.clg *
                       (self.fdust[f1]*self.fdust[f2]/self.feff**2)**self.beta_g *
                       self._PlanckFunctionRatio(self.fdust[f1],19.6) * self._PlanckFunctionRatio(self.fdust[f2],19.6) *
                       self._Flux2TempRatio( self.fdust[f1]) * self._Flux2TempRatio( self.fdust[f2])
                       )
        return np.array(dl)


#Synchrotron (TE,EE)
class Synchrotron(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "synchrotron"

        self.clsp = np.zeros( lmax+1)
        ell = np.arange( 2, lmax+1)
        self.clsp[2:] = (ell/500)**(-0.7)

    def compute_dl(self, pars):
        Async = 0
        if self.mode == "TE": Async = pars["a_ste"]
        if self.mode == "EE": Async = pars["a_see"]
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( Async * self.clsp *
                       (self.fsyn[f1]/self.feff * self.fsyn[f2]/self.feff)**self.alpha_gs *
                       self._Flux2TempRatio( self.fsyn[f1]) * self._Flux2TempRatio( self.fsyn[f2])
                       )
        return np.array(dl)


#thermal SZ (TT)
class tSZ(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "tsz"

        self.cltsz = np.zeros( lmax+1)
        self.cltsz[2:] = np.loadtxt( filename, unpack=True)[1,:lmax-1]
        self.cltsz /= 5.59550 #This normalizes the tSZ spectrum to 1 at l=3000

    def compute_dl(self, pars):
        f0 = self._sz_func( self.feff)
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_tsz"] * self.cltsz * self._sz_func(self.fsz[f1])/f0 * self._sz_func(self.fsz[f2])/f0 )
            
        return np.array(dl)

#kinetic SZ (TT)
class kSZ(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "ksz"

        self.clksz = np.zeros( lmax+1)
        self.clksz[2:] =np.loadtxt( filename, unpack=True)[1,:lmax-1]
        self.clksz /= 1.51013 #This normalizes the kSZ spectrum to 1 at l=3000
        
    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_ksz"] * self.clksz)
            
        return np.array(dl)

#tSZ-CIB correlation (TT)
class tSZxCIB(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True, survey=""):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "tszxcib"

        self.clszcib = np.zeros( lmax+1)
        self.clszcib[2:] =np.loadtxt( filename, unpack=True)[1,:lmax-1]
        
    def compute_dl(self, pars):
        f0 = self._sz_func( self.feff)
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( -2 * np.sqrt(pars["a_c"]*pars["a_tsz"]) * pars["xi"] * self.clszcib *
                       ( self.fdust[f1]**pars["beta_c"]*self._sz_func(self.fsz[f2])*self._PlanckFunctionRatio(self.fdust[f1],9.7)*self._Flux2TempRatio(self.fdust[f1]) +
                         self.fdust[f2]**pars["beta_c"]*self._sz_func(self.fsz[f1])*self._PlanckFunctionRatio(self.fdust[f2],9.7)*self._Flux2TempRatio(self.fdust[f2])
                         ) / (2*self.feff**pars["beta_c"]*f0)
                       )
            
        return np.array(dl)

