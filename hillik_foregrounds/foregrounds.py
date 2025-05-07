# FOREGROUNDS CMB LIK
import astropy.io.fits as fits
import numpy as np
import itertools
from cobaya.log import HasLogger, LoggedError
from scipy import constants


T_CMB = 2.72548
k_b = constants.k #1.3806503e-23
h_pl = constants.h #6.626068e-34


#Temp Antenna conversion
def dBdT( f):
    """
    Computes the conversion factor :math:`\frac{\partial B_{\nu}}{\partial T}`
    from CMB thermodynamic units to differential source intensity.

    :param f: frequency in  GHz
    """
    nu  = f*1e9
    x   = h_pl*nu /(k_b*T_CMB)
    return nu**4 * np.exp(x) / np.expm1(x)**2.

def f_Planck( f, T):
    """
    Computes Planck spectrum

    :param f: frequency in GHz
    :param T: Temperature in K
    """
    nu = f*1e9
    x  = h_pl*nu /(k_b*T)
    return nu**3 / np.expm1(x)

def f_tsz( f):
    """
    Thermal Sunyaev-Zel'dovich in K_CMB

    :param f: frequency in GHz
    """
    nu = f*1e9
    x = h_pl*nu/(k_b*T_CMB)
    return x/np.tanh(x/2.) - 4



# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(HasLogger):
    """
    Class of foreground model in spectra domain
    Units: Dl in muK^2
    Return the foreground model in Dl for all required cross-correlations
    """

    #frequency reference for fg amplitudes
    # ACT: 150 dust, 150 tSZ
    # SPT: 220 dust, 143 tSZ
    # SPT3G: 150 dust, 143 tSZ
    nu0 = 150.
    lnorm = 3000

    def __init__(self, lmax, cross_freq, mode="TT", survey="", **kwargs):
        """
        Create model for foreground

        Input:
            lmax: maxmimu multipole
            cross_freq: list of cross-frequencies
            mode: 'TT','TE', or 'EE'
            survey: name of the survey (avail. 'PLK','ACT','SPT','SPT3G')
        Options:
            filename: template for spectrum
            filenames: list of two template (or SZxCIB)
        """
        self.mode = mode
        self.lmax = lmax
        self.survey = survey

        #depend on the model
        self.name = None
        self.dlfg = None
        self.sed  = None

        self._cross_frequencies = cross_freq
        self.ncross = len(cross_freq)
        self._freqs = np.unique(list(itertools.chain(*cross_freq)))
        
        self.set_logger()

    def _gen_dl_powerlaw( self, alpha):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = max(self.lmax,self.lnorm)
        ell = np.arange( 2, lmax+1)

        template = np.zeros( lmax+1)
        template[np.array(ell,int)] = ell*(ell+1)/2/np.pi * ell**(alpha)

        #normalize Dl
        template = template / template[self.lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """

        #read dl template
        l,data = np.loadtxt( filename, unpack=True)
        self.log.debug( "Template: {}".format(filename))

        if max(l) < self.lmax:
            self.log.info( "WARNING: template {} has lower lmax (filled with 0)".format(filename))
        template = np.zeros( max(self.lmax,int(max(l))) + 1)
        template[np.array(l,int)] = data

        #normalize Dl
        if self.lnorm < max(l):
            template = template / template[self.lnorm]
        else:
            raise ValueError( f"WARNING: template {filename} cannot be normalized (l0={self.lnorm}")
        
        return template[:self.lmax+1]

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass

    def get_sed( self):
        """
        Return frequency model
        """
        return self.sed

    def get_dlfg( self):
        """
        Return spectra shape
        """
        return self.dlfg

        


class SED(HasLogger):
    """
    Class for computing SED of foreground components in muK_CMB
    It takes into account instrumental effects by either use effective frequencies,
    integrate over the bandpass, or include beam chromaticity (see ACT-DR6)
    """
    name = ""
    
    def __init__(self, **kwargs):
        """
        Init SED for foregrounds
        Can be of three types:
            - integrate beam chromaticity over the bandpass (fg_Ratio,beams=dict(freq={nu,beam,bandpass}))
            - bandpass integration (fg_Ratio,bandpass=dict(freq={nu,transmission}))
            - effective frequency (fg_Ratio,nu_eff=dict(freq=feff))
        """

        amp = {}

        if 'beams' in kwargs:
            self._beams = kwargs['beams']
            self._sed = self._beam_chromaticity
            self._bp_shifts = {m:0. for m in self._beams}
            
        elif 'bandpass' in kwargs:
            self_.bandpass = kwargs['bandpass']
            self._sed = self._bandpass_integration

        elif 'feff' in kwargs:
            self._feff = kwargs['feff'][self.name]
            self._sed = self._nu_eff
        else:
            self._sed = self._nu_default

    #DEFINE SED FOR THE SKY COMPONENT
    def fgRatio( nu, nu0, **kwargs):
        pass

    def set_bandpass_shifts( self, bp_shifts):
        for k,v in bp_shifts.items():
            self._bp_shifts[k] = v
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def eval( self, exps, **kwargs):
        return self._sed( exps, **kwargs)

    def _beam_chromaticity( self, exps, **kwargs):
        amp = {}
        for f in exps:
            bpl = self._beams[f]

            nu = bpl['nu'] + self._bp_shifts[f]
            rl = bpl['bandpass']*dBdT(nu)*bpl['beam']
            rl /= np.trapz(rl,nu)[...,np.newaxis]
            
            fl = rl*self.fgRatio(nu, **kwargs)
            amp[f] = np.trapz(fl,nu)

        return amp

    def _bandpass_integration( self, exps, **kwargs):
        amp = {}
        for f in exps:
            bp = self._bandpass[f]
            nu = bp['nu'] + self._bp_shifts[f]
            U = np.trapz(bp['transmission']*dBdT(nu)*self.fgRatio(bp['nu'],**kwargs), nu)
            D = np.trapz(bp['transmission']*dBdT(nu), nu)
            amp[f] = U/D
        return amp

    def _nu_eff( self, exps, **kwargs):
        amp = {}
        for f in exps:
            amp[f] = self.fgRatio(self._feff[f],**kwargs)
        return amp

    def _nu_default( self, exps, **kwargs):
        amp = {}
        for f in exps:
            amp[f] = self.fgRatio( f,**kwargs)
        return amp
        


#frequency reference for fg amplitudes
# ACT: 150 dust, 150 tSZ
# SPT: 220 dust, 143 tSZ
# SPT3G: 150 dust, 143 tSZ

class TSZ(SED):
    name = "tsz"
    def fgRatio( self, nu, nu0=150):
        return f_tsz(nu)/f_tsz(nu0)

class CIB(SED):
    name = "cib"
    def fgRatio( self, nu, nu0=150, beta=1.75, T=25): #T=9.7 ?!?
        return (nu/nu0)**beta * (f_Planck(nu,T)/f_Planck(nu0,T)) / ( dBdT(nu)/dBdT(nu0) )
    
class DUST(SED):
    name = "dust"
    def fgRatio( self, nu, nu0=150, beta=1.5, T=19.6):
        return (nu/nu0)**beta * (f_Planck(nu,T)/f_Planck(nu0,T)) / ( dBdT(nu)/dBdT(nu0) )
    
class RADIO(SED):
    name = "radio"
    def fgRatio( self, nu, nu0=150, beta=-0.7):
        return (nu/nu0)**beta / ( dBdT(nu)/dBdT(nu0) )

class SYNC(SED):
    name = "sync"
    def fgRatio( self, nu, nu0=150, beta=-0.7):
        return (nu/nu0)**beta / ( dBdT(nu)/dBdT(nu0) )



# ------------------------------------------------------------------------------------------------




# Point Sources (for TT)
class ps(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey=""):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "PS"
        self.dlfg = self._gen_dl_powerlaw(0.)

    def compute_dl(self, pars):
        if self.mode == "TT":
            dl_ps = []
            for f1, f2 in self._cross_frequencies:
                dl_ps.append( pars[f"{self.survey}_ps_{f1}x{f2}"] * self.dlfg)
            return np.array(dl_ps)
        else:
            return 0.


# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "PS radio"
        self.dlfg = self._gen_dl_powerlaw(0.)
        self.sed = RADIO( **kwargs)

    def compute_dl(self, pars):
        dl = []
        self.sed.set_bandpass_shifts( {f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=self.nu0, beta=pars['beta_radio'])
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dlfg * sed[f1] * sed[f2] )

        if self.mode == "TT":
            return pars[f"{self.survey}_radio_ps"] * np.array(dl)
        else:
            return 0.


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "PS dusty"
        self.dlfg = self._gen_dl_powerlaw(0.)
        self.sed  = CIB( **kwargs)

    def compute_dl(self, pars):
        dl = []
        self.sed.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=self.nu0, beta=pars['beta_dusty'], T=pars.get('T_cib',25.))
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dlfg * sed[f1] * sed[f2] )

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
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "Dust"
        self.lnorm = kwargs.get('lnorm',200)
        self.sed = DUST( **kwargs)
        
    def compute_dl(self, pars):
        dl = []

        nu0 = 353 if self.survey == "PLK" else 150
        self.sed.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=nu0, beta=pars[f'{self.survey}_beta_dust{self.mode}'], T=pars.get('T_dust',19.6))

        for xf, (f1, f2) in enumerate(self._cross_frequencies):

            if self.survey == "PLK":
                #rescale PLK for each combination of mask
                alpha = pars[f'{self.survey}_alpha_dust{max(f1,f2)}{self.mode}']
                ad    = pars[f'{self.survey}_Adust{max(f1,f2)}{self.mode}']
            else:
                alpha = pars[f'{self.survey}_alpha_dust{self.mode}']
                ad    = pars[f'{self.survey}_Adust{self.mode}']            
            dlg = self._gen_dl_powerlaw( alpha)

            ell = np.arange( 2,self.lmax+1)
            dlg = np.zeros( self.lmax+1)
            dlg[ell] = (ell/self.lnorm)**(2+alpha)
            
            dl.append( ad * dlg * sed[f1] * sed[f2] )

        return np.array(dl)


#Dust amplitudes
class dust_amplitude(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "Dust Amplitudes"
        self.lnorm = 200
        self.dlfg = np.zeros( lmax+1)

        ell = np.arange( 2, lmax+1)
        alpha_dust = -2.5 if mode == "TT" else -2.4
        self.dlfg = self._gen_dl_powerlaw( alpha_dust)
    
    def compute_dl(self, pars):
        if   self.mode == "TT": ad1,ad2 = f'{self.survey}_dustT',f'{self.survey}_dustT'
        elif self.mode == "TE": ad1,ad2 = f'{self.survey}_dustT',f'{self.survey}_dustP'
        elif self.mode == "ET": ad1,ad2 = f'{self.survey}_dustP',f'{self.survey}_dustT'
        elif self.mode == "EE": ad1,ad2 = f'{self.survey}_dustP',f'{self.survey}_dustP'

        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( pars[ad1+f"_{f1}"] * pars[ad2+f"_{f2}"] * self.dlfg)

        return np.array(dl)


# Synchrotron model
class sync(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "Synchrotron"
        self.lnorm = 200

        alpha_syn = -2.5  #Cl template power-law TBC
        self.dlfg = self._gen_dl_powerlaw( alpha_syn)
        self.sed = SYNC( **kwargs)

    def compute_dl(self, pars):
        dl = []
        self.sed.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=self.nu0, beta=beta_syn)
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dlfg * sed[f1] * sed[f2] )
    
        if self.mode == "TT":
            return pars[f"{self.survey}_AsyncT"] * np.array(dl)
        elif self.mode == "EE":
            return pars[f"{self.survey}_AsyncP"] * np.array(dl)
        else:
            return 0.


# CIB clustered (one spectrum for all freqs)
class cib(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "clustered CIB"

        if 'filename' in kwargs:
            self.dlfg = self._read_dl_template( kwargs['filename'])
        else:
            alpha_cib = -1.3
            self.dlfg = self._gen_dl_powerlaw( alpha_cib)
        self.sed = CIB( **kwargs)

    def compute_dl(self, pars):
        dl = []
        self.sed.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=self.nu0, beta=pars['beta_cib'],T=pars.get('T_cib',25.))
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dlfg * sed[f1] * sed[f2] )

        if self.mode == "TT":
            return pars["Acib"] * np.array(dl)
        else:
            return 0.


#thermal SZ (one spectrum for all freqs)
class tsz(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "tSZ"

        if 'filename' not in kwargs:
            raise ValueError( f"Missing SZ Cl shape")
            
        self.dlfg = self._read_dl_template( kwargs['filename'])
        self.sed  = TSZ( **kwargs)

    def compute_dl(self, pars):

        dl_sz = []
        self.sed.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        sed = self.sed( self._freqs, nu0=self.nu0)
        for f1, f2 in self._cross_frequencies:
            dl_sz.append( self.dlfg * sed[f1] * sed[f2] )

        if self.mode == "TT":
            return pars["Atsz"] * np.array(dl_sz)
        else:
            return 0.


#kinetic SZ (one spectrum for all freqs)
class ksz(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "kSZ"

        if 'filename' not in kwargs:
            raise ValueError( f"Missing SZ Cl shape")
            
        self.dlfg = self._read_dl_template( kwargs['filename'])
        self.dl_ksz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(self.dlfg)
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Aksz"] * self.dl_ksz
        else:
            return 0.



# SZxCIB model (one spectrum for all freqs)
class szxcib(fgmodel):
    def __init__(self, lmax, cross, mode="TT", survey="", **kwargs):
        super().__init__(lmax, cross, mode=mode, survey=survey)
        self.name = "SZxCIB"

        if 'filename' in kwargs:
            self.dlfg = self._read_dl_template(kwargs['filename'])
        elif "filenames" in kwargs:
            self.dlfg = self._read_dl_template(kwargs["filenames"][0])*self._read_dl_template(kwargs["filenames"][1])
        else:
            raise ValueError( f"Missing template for SZxCIB  for {self.survey}")

        self.sed_tsz = TSZ( **kwargs)
        self.sed_cib = CIB( **kwargs)

    def compute_dl(self, pars):
        dl_szxcib = []
        self.sed_tsz.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        self.sed_cib.set_bandpass_shifts({f:pars.get(f'{self.survey}_band_shift_{f}',0) for f in self._freqs})
        tsz = self.sed_tsz( self._freqs, nu0=self.nu0)
        cib = self.sed_cib( self._freqs, nu0=self.nu0, beta=pars['beta_cib'], T=pars.get('T_cib',25))
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.dlfg * np.sqrt(pars["Acib"]*pars["Atsz"]) * ( tsz[f2]*cib[f1] + tsz[f1]*cib[f2] ) )

        if self.mode == "TT":
            return -1. * pars["xi"] * np.array(dl_szxcib)
        else:
            return 0.
