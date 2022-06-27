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
        "SPT": {95:96.55, 150:152.26, 220:220.1},  #norm 143 !
        "ACTw": {98: 98.4, 150: 150.1},
        "ACTd": {98: 98.4, 150: 150.1},
        }

    fdust = {
        "PLK": {100:105.25, 143:148.23, 217:229.1, 353:372.19}, #alpha=4 from [Planck 2013 IX]
        "SPT": {95:96.89, 150:153.37, 220:221.6}, #norm dust: 220GHz
        "ACTw": {98: 98.8, 150: 151.2},
        "ACTd": {98: 98.8, 150: 151.2},
        }    
#        "PLK": {100:106.16, 143:149.56, 217:230.848, 353:372.},

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
    
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.name = None
        self.survey = survey
        self.lnorm = lnorm #to normalize templates in cl

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
            #Use auto spectra and combine TE with ET (e.g. ACT, SPT)
            self._cross_frequencies = list(itertools.combinations_with_replacement(freqs, 2))
        else:
            #Remove auto-spectra and use separate TE and ET (e.g. Planck)
            self._cross_frequencies = list(itertools.combinations(freqs, 2))

        self.set_logger()
        pass

    def _gen_dl_powerlaw( self, alpha):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = self.lmax if self.lnorm is None else max(self.lmax,self.lnorm)
        ell = np.arange( 2, lmax+1)

        template = np.zeros( lmax+1)
        template[np.array(ell,int)] = ell*(ell+1)/2/np.pi * ell**(alpha)

        #normalize l=3000
        if self.lnorm is not None:
            template = template / template[self.lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """

        #read dl template
        l,data = np.loadtxt( filename, unpack=True)

        template = np.zeros( max(self.lmax,int(max(l))) + 1)
        template[np.array(l,int)] = data

        #normalize l=3000
        if self.lnorm is not None:
            template = template / template[self.lnorm] #* self.lnorm*(self.lnorm+1)/2/np.pi
        
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
        self.dltemp = self._gen_dl_powerlaw(0.)

    def compute_dl(self, pars):
        if self.mode == "TT":
            dl_ps = []
            for f1, f2 in self._cross_frequencies:
                dl_ps.append( pars[f"Aps_{self.survey}_{f1}x{f2}"] * self.dltemp)
            return np.array(dl_ps)
        else:
            return 0.



#Galactic Dust model
#ACT: -0.6 for TT, -0.4 for TE
#SPT: -1.2 for TT (cirrus but very low amplitude) ?!?
#Planck: -0.63 for TT, -0.4 for TE
class dust(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=200):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "Dust Model"

        if filename is None:
            alpha_dust = -2.5 if mode == "TT" else -2.4
            self.dlg = self._gen_dl_powerlaw( alpha_dust)
        else:
            self.dlg = self._read_dl_template( filename)
        
    def compute_dl(self, pars):
        Ad = []
        for f1, f2 in self._cross_frequencies:
            Ad.append(pars[f"Adust_{self.survey}_{f1}"]*pars[f"Adust_{self.survey}_{f2}"])
#            if self.mode == "TT": Ad.append(pars[f"Adust_{self.survey}_{f1}T"]*pars[f"Adust_{self.survey}_{f2}T"])
#            if self.mode == "TE": Ad.append(pars[f"Adust_{self.survey}_{f1}T"]*pars[f"Adust_{self.survey}_{f2}P"])
#            if self.mode == "ET": Ad.append(pars[f"Adust_{self.survey}_{f1}P"]*pars[f"Adust_{self.survey}_{f2}T"])
#            if self.mode == "EE": Ad.append(pars[f"Adust_{self.survey}_{f1}P"]*pars[f"Adust_{self.survey}_{f2}P"])
        
        return np.array(Ad)[:, None] * self.dlg


# CIB clustered (one spectrum for all freqs)
class cib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "clustered CIB"

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw( alpha_cib)
        else:
            self.dl_cib = self._read_dl_template( filename)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio(self.fdust[self.survey][f1],self.feff,pars['beta_cib'])
                       * self._cibRatio(self.fdust[self.survey][f2],self.feff,pars['beta_cib'])
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
        
        self.dl_sz = []
        sz_tmpl = self._read_dl_template( filename)
        
        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append( sz_tmpl
                               * self._tszRatio(self.fsz[self.survey][f1],self.feff)
                               * self._tszRatio(self.fsz[self.survey][f2],self.feff)
                               )
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Atsz"] * self.dl_sz
        else:
            return 0.


#kinetic SZ (one spectrum for all freqs)
class ksz(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "kSZ"

        self.dl_ksz = []
        ksz_tmpl = self._read_dl_template( filename)
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksz_tmpl)
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Aksz"] * self.dl_ksz
        else:
            return 0.



# SZxCIB model (one spectrum for all freqs)
class szxcib(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False, survey="", filename=None, lnorm=3000):
        super().__init__(lmax, freqs, mode=mode, auto=auto, survey=survey, lnorm=lnorm)
        self.name = "SZxCIB"
        
        self.x_tmpl = self._read_dl_template(filename)
    
    def compute_dl(self, pars):
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.x_tmpl * (
                self._tszRatio(self.fsz[self.survey][f2],150) * self._cibRatio(self.fdust[self.survey][f1], 150, pars['beta_cib']) +
                self._tszRatio(self.fsz[self.survey][f1],150) * self._cibRatio(self.fdust[self.survey][f2], 150, pars['beta_cib'])
                ) / 2
            )

        if self.mode == "TT":
            return -2. * np.sqrt(pars["Acib"]*pars["Atsz"]) * pars["xi"] * np.array(dl_szxcib)
        else:
            return 0.
