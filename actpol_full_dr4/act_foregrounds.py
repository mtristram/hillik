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

    def _sz_func( freq):
        """
        Freq in GHz
        """

        nu = freq*1e9
        xx=hpl*nu/(k_b*t_cmb)
        return xx*(1/tanh(xx/2.d0)) - 4

    #t_eff = 9.7
    #t_eff = 19.6
    def _PlanckFunctionRatio(freq,t_eff,fe=150):
        nu  = freq*1e9
        nu0 = fe*1e9
        xx  = h_pl*nu /(k_b*t_eff)
        xx0 = h_pl*nu0/(k_b*t_eff)

        return (nu/nu0)**3.*(exp(xx0)-1.)/(exp(xx)-1.)

    def _Flux2TempRatio(freq,fe=150):
        # rescaled to 150 GHz
        nu  = freq*1e9
        nu0 = fe*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        xx0 = h_pl*nu0/(k_b*t_cmb)
        return (nu0/nu)**4.*(exp(xx0)/exp(xx))*((exp(xx)-1.)/(exp(xx0)-1.))**2.
        

    def __init__(self, lmax, freqs, mode="TT", auto=False):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        # Build the list of cross frequencies
        nfreq = len(freqs)
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(range(nfreq), 2)
            if auto
            else itertools.combinations(range(nfreq), 2)
        )
        self.set_logger()
        pass

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass


# ------------------------------------------------------------------------------------------------



# Point Sources
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "PS"
        # Amplitudes of the point sources power spectrum per xfreq
        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / 2.0 / np.pi

    def compute_dl(self, pars):
        nfreq = len(self.freqs)
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            freq1 = self.freqs[f1]
            freq2 = self.freqs[f2]
            dl_ps.append( pars["Aps_{}x{}".format(freq1,freq2)] * 1e-6 * self.ll2pi)

        return np.array(dl_ps)

#CIB clustered (TT)
class CIBclustered(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True):
        super().__init__(lmax, freqs, auto)
        self.name = "clustered_cib"

        ell = np.arange( lmax+1)
#        self.clc=(ell*(ell+1)/3000/3001)*(ell/3000.d0)**(-1.2)
        self.clc = np.loadtxt( filename, unpack=True)[1,:lmax+1]
        
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
    def __init__(self, lmax, freqs, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
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


#RADIO poisson (TT, TE, EE)
class RADIOpoisson(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "poisson_radio"
        self.mode = mode
        ell = np.arange( lmax+1)
        self.clp=(ell*(ell+1)/3000/3001)

        
    def compute_dl(self, pars):
        Aps = 0
        if self.mode is "TT": Aps = pars['a_s'] 
        if self.mode is "TE": Aps = pars['a_tps'] 
        if self.mode is "EE": Aps = pars['a_ps'] 
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( Aps * self.clp *
                       (self.fsyn[f1]*self.fsyn[f2]/self.feff**2)**self.alpha_s *
                       self._Flux2TempRatio( self.fsyn[f1]) * self._Flux2TempRatio( self.fsyn[f2])
                       )
        return np.array(dl)

#Galactic Dust (TT,TE,EE)
class GalacticDust(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "galactic dust"

        ell = np.arange( lmax+1)
        if mode is "TT":
            self.clg = (ell/500)**(-0.6) #clgt
        else:
            self.clg = (ell/500)**(-0.4) #clgp

        
    def compute_dl(self, pars):
        Adust = 0
        if mode is "TT": Adust = pars["a_gd"]
        if mode is "TE": Adust = pars["a_gted"]
        if mode is "EE": Adust = pars["a_geed"]
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( Adust * self.clgt *
                       (self.fdust[f1]*self.fdust[f2]/self.feff**2)**self.beta_g *
                       self._PlanckFunctionRatio(self.fdust[f1],19.6) * self._PlanckFunctionRatio(self.fdust[f2],19.6) *
                       self._Flux2TempRatio( self.fdust[f1]) * self._Flux2TempRatio( self.fdust[f2])
                       )
        return np.array(dl)


#Synchrotron (TE,EE)
class Synchrotron(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "synchrotron"

        ell = np.arange( lmax+1)
        self.clsp = (ell/500)**(-0.7)

        
    def compute_dl(self, pars):
        Async = 0
        if mode is "TE": Async = pars["a_ste"]
        if mode is "EE": Async = pars["a_see"]
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_see"] * self.clsp *
                       (self.fsyn[f1]/self.feff * self.fsyn[f2]/self.feff)**self.alpha_gs *
                       self._Flux2TempRatio( self.fsyn[f1]) * self._Flux2TempRatio( self.fsyn[f2])
                       )
        return np.array(dl)


#thermal SZ (TT)
class tSZ(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "tsz"

        ell = np.arange( lmax+1)
        self.cltsz = np.loadtxt( filename, unpack=True)[1,:lmax+1]
        self.cltsz /= 5.59550 #This normalizes the tSZ spectrum to 1 at l=3000

        
    def compute_dl(self, pars):
        f0 = self.sz_func( self.feff)
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_tsz"] * self.cltsz * self.sz_func(self.fsz[f1])/f0 * self.sz_func(self.fsz[f2])/f0 )
            
        return np.array(dl)

#kinetic SZ (TT)
class kSZ(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "ksz"

        ell = np.arange( lmax+1)
        self.clksz =np.loadtxt( filename, unpack=True)[1,:lmax+1]
        self.cltsz /= 1.51013 #This normalizes the kSZ spectrum to 1 at l=3000
        
    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars["a_ksz"] * self.clksz)
            
        return np.array(dl)

#tSZ-CIB correlation (TT)
class tSZxCIB(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=True):        
        super().__init__(lmax, freqs, auto)
        self.name = "tszxcib"

        ell = np.arange( lmax+1)
        self.clszcib =np.loadtxt( filename, unpack=True)[1,:lmax+1]
        
    def compute_dl(self, pars):
        f0 = self.sz_func( self.feff)
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( -2 * sqrt(pars["a_c"]*pars["a_tsz"]) * pars["xi"] * self.clszcib *
                       ( self.fdust[f1]**pars["beta_c"]*self.sz_fun(self.fsz[f2])*self._PlanckFunctionRatio(self.fdust[f1],9.7)*self._Flux2TempRatio(self.fdust[f1]) +
                         self.fdust[f2]**pars["beta_c"]*self.sz_fun(self.fsz[f1])*self._PlanckFunctionRatio(self.fdust[f2],9.7)*self._Flux2TempRatio(self.fdust[f2])
                         ) / (2*self.feff**pars["beta_c"]*f0
                       )
            
        return np.array(dl)

