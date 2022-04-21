# FOREGROUNDS CMB LIK
import astropy.io.fits as fits
import numpy as np
import itertools
from cobaya.log import HasLogger, LoggedError


t_cmb = 2.725
k_b = 1.3806503e-23
h_pl = 6.626068e-34

#Parameter names
#A{fg}_{survey}_{freq(xfreq)}

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
    fsz = {
        "PLK": {100:100.2, 143: 143, 217: 222},
        "SPT": {95:96.55, 150:152.26, 220:220.1},
        "ACTw": {98: 98.4, 150: 150.1},
        "ACTd": {98: 98.4, 150: 150.1},
        }

    fdust = {
        "PLK": {100:106.16, 143:149.56, 217:230.848, 353:372.},
        "SPT": {95:96.89, 150:153.37, 220:221.6},
        "ACTw": {98: 98.8, 150: 151.2},
        "ACTd": {98: 98.8, 150: 151.2},
        }
    
#    fsz   = {98: 98.4, 150: 150.1} #ACT
#    fdust = {98: 98.8, 150: 151.2} #ACT

    #SPT: only one serie of effective frequencies ?
#    fsz   = {90:96.55, 150:152.26, 220:220.1} #norm 143 !
#    fdust = {90:96.89, 150:153.37, 220:221.6} #norm dust: 220GHz

    #Planck: need to compute...
#    fsz   = {100:100.2, 143: 143, 217: 222}
#    fdust = {100:106.16, 143:149.56, 217:230.848, 353:372.}

    #frequency reference for fg amplitudes
    feff = 150

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
        return self._f_tsz(f)/self._f_tsz(f0)

    def _cibRatio( self, f, f0, beta, T=9.7):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )
    
    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )
    
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.name = None
        self.survey = survey

        #check effective freqs
        if self.survey not in self.fsz.keys():
            raise ValueError( f"Missing SZ effective frequency for {self.survey}")
        if self.survey not in self.fdust.keys():
            raise ValueError( f"Missing DUST effective frequency for {self.survey}")

        for f in freqs:
            if f not in self.fsz[self.survey].keys():
                raise ValueError( f"Missing SZ effective frequency for {f}")
            if f not in self.fdust[self.survey].keys():
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
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "PS"
        # Amplitudes of the point sources power spectrum per xfreq
        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / 2.0 / np.pi

    def compute_dl(self, pars):
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append( pars[f"Aps_{self.survey}_{f1}x{f2}"] * 1e-6 * self.ll2pi)

        return np.array(dl_ps)



#Galactic Dust model
#ACT: -0.6 for TT, -0.4 for TE
#SPT: -1.2 for TT (cirrus but very low amplitude) ?!?
#Planck: -0.63 for TT, -0.4 for TE
class dust(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "Dust Model"
        self.dlg = np.zeros( lmax+1)

        if filename is None:
            ell = np.arange( 2, lmax+1)
            alpha_dust = -2.6 if mode == "TT" else -2.4
            self.dlg[2:] = (ell)**(alpha_dust+2)
        else:
            data = fits.getdata( filename)
            self.dlg[data.ell] = data.dl
        
    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dlg
                       * self._dustRatio(self.fdust[self.survey][f1],353, beta=pars["beta_dust"], T=pars["T_dust"])
                       * self._dustRatio(self.fdust[self.survey][f2],353, beta=pars["beta_dust"], T=pars["T_dust"])
                       )
        return pars[f'Adust_{self.survey}'] * np.array(dl)


# CIB clustered (one spectrum for all freqs)
class cib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "clustered CIB"
        self.dl_cib = np.zeros(lmax + 1)

        if filename is None:
            ell = np.arange( 2, lmax+1)
            alpha_cib = -1.3
            self.dlg[2:] = (ell)**(alpha_cib+2)
        else:
            l,dl = np.loadtxt( filename, unpack=True)
            self.dl_cib[np.array(l[l<=lmax],int)] = dl[l<=lmax]

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio(self.fdust[self.survey][f1],self.feff,pars['beta_cib'])
                       * self._cibRatio(self.fdust[self.survey][f2],self.feff,pars['beta_cib'])
                       )
        
        return pars["Acib"] * np.array(dl)


#thermal SZ (one spectrum for all freqs)
class tsz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "tSZ"
        
        self.dl_sz = []
        l,dl = np.loadtxt( filename, unpack=True)
        sz_tmpl = np.zeros( lmax+1)
        sz_tmpl[np.array(l[l<=lmax],int)] = dl[l<=lmax]
        
        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append( sz_tmpl
                               * self._tszRatio(self.fsz[self.survey][f1],self.feff)
                               * self._tszRatio(self.fsz[self.survey][f2],self.feff)
                               )
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Atsz"] * self.dl_sz


#kinetic SZ (one spectrum for all freqs)
class ksz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "kSZ"

        self.dl_ksz = []
        l,dl = np.loadtxt( filename, unpack=True)
        ksz_tmpl = np.zeros( lmax+1)
        ksz_tmpl[np.array(l[l<=lmax],int)] = dl[l<=lmax]
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksz_tmpl)
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        return pars["Aksz"] * self.dl_ksz



# SZxCIB model (one spectrum for all freqs)
class szxcib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey)
        self.name = "SZxCIB"
        
        l,dl = np.loadtxt( filename, unpack=True)
        self.x_tmpl = np.zeros( lmax+1)
        self.x_tmpl[np.array(l[l<=lmax],int)] = dl[l<=lmax]
    
    def compute_dl(self, pars):
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.x_tmpl * (
                self._f_tsz(self.fsz[self.survey][f2]) * self._cibRatio(self.fdust[self.survey][f1], 150, pars['beta_cib']) +
                self._f_tsz(self.fsz[self.survey][f1]) * self._cibRatio(self.fdust[self.survey][f2], 150, pars['beta_cib'])
                ) / 2
            )
        
        return -2. * np.sqrt(pars["Acib"]*pars["Atsz"]) * pars["xi"] * np.array(dl_szxcib)

