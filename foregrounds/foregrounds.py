# FOREGROUNDS CMB LIK
import astropy.io.fits as fits
import numpy as np
import itertools
from cobaya.log import HasLogger, LoggedError


t_cmb = 2.725
k_b = 1.3806503e-23
h_pl = 6.626068e-34


# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(HasLogger):
    """
    Class of foreground model for the Hillik likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    #equivalent frequencies (experiment dependant)
    fsz   = {98: 98.4, 150: 150.1} #ACT
    fdust = {98: 98.8, 150: 151.2} #ACT

    #SPT: only one serie of effective frequencies ?
    #norm dust: 220GHz

    #Planck: need to compute...
    fsz   = {100:100.2, 143: 143, 217: 222}
    fdust = {}

    #frequency reference for fg amplitudes
    feff = 143

    def _f_tsz( self, freq):
        # Freq in GHz
        nu = freq*1e9
        xx=h_pl*nu/(k_b*t_cmb)
        return xx*( 1/np.tanh(xx/2.) ) - 4

    def _f_Planck( self, f, T):
        # Freq in GHz
        nu = f*1e9
        xx  = h_pl*nu /(k_b*T)
        return (nu**3.)/(np.exp(xx)-1.)

    #1/Flux2Temp
    def _dBdT(self, f):
        # Freq in GHz
        nu  = f*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        return (nu)**4 * np.exp(xx) / (np.exp(xx)-1.)**2.

    def _tszRatio( self, f, f0):
        return _f_tsz(f)/_f_tsz(f0)

    def _cibRatio( self, f, f0, beta, T=9.7):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def __init__(self, lmax, freqs, mode="TT", auto=False):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.name = None

        #check freqs
        for f in freqs:
            if f is not in fsz.keys():
                raise ValueError( f"Missing SZ effective frequency for {f}")
            if f is not in fdust.keys():
                raise ValueError( f"Missing Dust effective frequency for {f}")

        # Build the list of cross frequencies
        if auto:
            if mode == "TE":
                self._cross_frequencies = list(itertools.product(freqs, repeat=2))
            else:
                self._cross_frequencies = list(itertools.combinations_with_replacement(freqs, 2))
        else:
            if mode == "TE":
                self._cross_frequencies = list(itertools.permutations(freqs, 2))
            else:
                self._cross_frequencies = list(itertools.combinations(freqs, 2))

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
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append( pars["Aps_{}x{}".format(f1,f2)] * 1e-6 * self.ll2pi)

        return np.array(dl_ps)



#Galactic Dust model
#ACT: -0.6 for TT, -0.4 for TE
#SPT: -1.2 for TT ?!?
class dust_model(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=True):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "Dust Model"
        
        ell = np.arange( 2, lmax+1)
        alpha_dust = -0.6 if mode == "TT" else alpha_dust = -0.4
        
        self.clg = np.zeros( lmax+1)
        self.clg[2:] = (ell/500)**(alpha_dust)
        
    def compute_dl(self, pars):        
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.clg * self._dustRatio(self.fdust[f1],self.feff) * self._dustRatio(self.fdust[f2],self.feff) )
        
        return pars['Adust'] * np.array(dl)


# CIB clustered (one spectrum for all freqs)
class cib_template(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "clustered CIB"

        cldata = fits.getdata(filename) #Dl, units uK
        ell = np.array(cldata.ELL, int)
        self.dl_cib = np.zeros(max(ell) + 1)
        self.dl_cib[ell] = (cldata.CL1HALO + cldata.CL2HALO)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio(self.fdust[f1],self.feff,pars['beta_c'])
                       * self._cibRatio(self.fdust[f2],self.feff,pars['beta_c'])
                       )

        return pars["Acib"] * np.array(dl)[: lmax + 1]


#thermal SZ (one spectrum for all freqs)
class tsz_template(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "tSZ"

        self.dl_sz = []
        dldata = fits.getdata(filename) #Dl, units uK at 143GHz
        ell = np.array(dldata.ELL, int)
        tmpl = np.zeros(max(ell) + 1)
        for f1, f2 in self._cross_frequencies:
            tmpl[ell] = (dldata.CL1HALO + dldata.CL2HALO) * self._tszRatio(self.fsz[f1],self.feff) * self._tszRatio(self.fz[f2],self.feff)
            self.dl_sz.append(tmpl[: lmax + 1])
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Asz"] * self.dl_sz


#kinetic SZ (one spectrum for all freqs)
class ksz_template(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "kSZ"

        self.dl_ksz = []
        dltemp = fits.getdata(filename)  #Dl, units uK
        ell = np.array(dltemp.ELL, int)
        tmp = np.zeros(np.max(ell) + 1)
        for f1, f2 in self._cross_frequencies:
            tmp[ell] = dltemp.DELL
            self.dl_ksz.append(tmp[: lmax + 1])
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        return pars["Aksz"] * self.dl_ksz



# SZxCIB model (one spectrum for all freqs)
class szxcib_template(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "SZxCIB"
        
        self.dl_szxcib = []
        cldata = fits.getdata( filename) #Dl, muK
        ell = np.array(cldata.ELL, int)
        tmpl = np.zeros(max(ell) + 1)
        for f1, f2 in self._cross_frequencies:
            tmpl[ell] = cldata.CL * (
                self._szRatio(self.fsz[f2])*self._cibRatio(self.fdust[f1]) +
                self._szRatio(self.fsz[f1])*self._cibRatio(self.fdust[f2])
                ) / 2
            self.dl_szxcib.append(tmpl[: lmax + 1])
        self.dl_szxcib = np.array(self.dl_szxcib)
    
    def compute_dl(self, pars):
        return -2. * sqrt(pars["Acib"]*pars["Atsz"]) * pars["xi"] * self.dl_szxcib

