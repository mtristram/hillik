""".. module:: ACTpol_full_DR4

:Synopsis: Definition of python-native CMB likelihood for ACT likelihood.
Adapted from Fortran likelihood code
https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.cfm

full ACTPol spectra at 98x98, 98x150 and 150x150 GHz from 350 < l < 8000 measured during 2013-2016 in temperature and polarization

:Author: Matthieu Tristram

"""
import os
from typing import Optional, Sequence

import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

from . import act_foregrounds as fg

fg_list = {
    "ps": fg.ps,
    "cib_clustered": fg.CIBclustered,
    "cib_poisson": fg.CIBpoisson,
    "radio_poisson": fg.RADIOpoisson,
    "galactic_dust": fg.GalacticDust,
    "synchrotron": fg.Synchrotron,
    "tsz": fg.tSZ,
    "ksz": fg.kSZ,
    "szxcib": fg.tSZxCIB,
    }



class ACTPolLikelihood(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr4/likelihoods/actpolfull_dr4.01.tar.gz",
        "data_path": "actpol_full_dr4",
    }

    frequencies: Sequence[int] = [98, 150]

    data_folder: Optional[str] # = "actpol_full_dr4/likelihood"
    spec_filename: Optional[str] #'coadd_cl_15mJy_data_200124.txt'
    cov_filename: Optional[str]  #'coadd_cov_15mJy_200519.txt'
    bbl_filename: Optional[str]  #'coadd_bpwf_15mJy_191127_lmin2.txt'
    leakd_filename: Optional[str] #'leak_TE_deep_200519.txt'

#    normalizeSZ_143GHz: Optional[bool] = True
#    callFGprior: Optional[bool] = True
#    applyFTSprior: Optional[bool] = True

    #--------------------------------------------------------------
    # change these to include/exclude observables 
    #--------------------------------------------------------------
    # Options are tt only, te only, ee only or all
    # i.e., true-false-false, false-true-false, false-false-true, true-true-true
    use_tt: Optional[bool] = False  #TT only
    use_te: Optional[bool] = False  #TE only
    use_ee: Optional[bool] = False  #EE only

    use_deep: Optional[bool]
    use_wide: Optional[bool]

    #--------------------------------------------------------------
    #Settings (should not be altered)
    #--------------------------------------------------------------
    # general settings
    #--------------------------------------------------------------
    tt_lmax = 6000

    #----------------------------------------------------------------
    # likelihood terms from ACT data
    #----------------------------------------------------------------
    nnu      = 2   # number of frequencies
    nspectot = 10  #nspecf*nspec+2 TE for 90x150 and 150x90
    nspecf   = 3   #95x95, 95x150, 150
    nspectt  = 3   #TT
    nspecte  = 4   #TE
    nspecee  = 3   #EE
    nbintt = 52    #max nbins in ACT TT data  
    nbinte = 52    #max nbins in ACT TE data
    nbinee = 52    #max nbins in ACT EE data
    nbint  = 520   #total bins
    lmax_win = 7925 #ell max of the full window functions 
    bmax0  = 52     #number of bins in full window function
    b0=5           # setting bins discarded in TT (i.e., ell>600)
    b1=0           # bins discarded for TE
    b2=0           # bins discarded for EE

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, "packages_path", None)):
            raise LoggedError(
                self.log,
                "No path given to CIB_Likelihood data. Set the likelihood property 'path' or the common property '%s'.",
                _packages_path,
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        # check data_folder
        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

        # Update data_folder location
#        self.data_folder = os.path.join(self.data_folder, "data/actpol_full_dr4.01/data")

        #-----------------------------------------------
        #load spectrum
        #-----------------------------------------------

        # read bandpowers (1040)
        # check file before
        self.b_dat = np.loadtxt(os.path.join(self.data_folder, self.spec_filename), unpack=True)
        self.b_dat = self.b_dat[0:self.nbint]

        # Read windows (520,7924) check l ?
        # check file before
        self.win_func = np.zeros( (self.nbint, self.lmax_win+1) )
        win_func = np.loadtxt(os.path.join(self.data_folder, self.bbl_filename))
        nbint,lmax_win = np.shape(win_func)
        if nbint != self.nbint or lmax_win != self.lmax_win-1:
            raise LoggedError(self.log, "Wrong size for Window function (%s)" % self.bbl_filename)
        self.win_func[:,2:] = win_func

        # Read leakage (52)
        # check file before
        ell, self.l98, self.l150 = np.loadtxt(os.path.join(self.data_folder, self.leakd_filename),unpack=True)
        if len(ell) != self.nbinte:
            raise LoggedError(self.log, "Wrong size for Leakage template (%s)" % self.leakd_filename)

        #-----------------------------------------------
        #Read Covariance Matrix
        #-----------------------------------------------
        #covmat (1040,1040)
        self.covmat = np.loadtxt( os.path.join(self.data_folder, self.cov_filename))
        self.covmat = self.covmat[:self.nbint,:self.nbint]

        #check survey
        self.survey = ""
        if self.use_wide and self.use_deep:
            raise LoggedError(self.log, "Choose survey DEEP or WIDE, not both")
        if self.use_wide: self.survey = "wide"
        if self.use_deep: self.survey = "deep"

        #check modes
        if self.use_tt + self.use_te + self.use_ee not in [1,3]:
            raise LoggedError(self.log, "Usage: TT, EE, TE or TT+TE+EE")

        #cut lmin TT
        for s in range(self.nspectt):
            ishift = s*self.nbintt
            self.covmat[ishift:ishift+self.b0,ishift:ishift+self.b0] = np.identity(self.b0)*1e10

        #cut lmin TE
        for s in range(self.nspecte):
            ishift = self.nspectt*self.nbintt + s*self.nbinte
            self.covmat[ishift:ishift+self.b1,ishift:ishift+self.b1] = np.identity(self.b1)*1e10

        #cut lmin EE
        for s in range(self.nspecee):
            ishift = self.nspectt*self.nbintt + self.nspecte*self.nbinte + s*self.nbinee
            self.covmat[ishift:ishift+self.b2,ishift:ishift+self.b2] = np.identity(self.b2)*1e10

        # Init foreground model
        self.fgs = {'tt':[],'te':[],'ee':[]}
        for tag,is_used in {"tt":self.use_tt,"te":self.use_te,"ee":self.use_ee}.items():
            if is_used:
                for name in self.foregrounds[tag.upper()].keys():
                    if name not in fg_list.keys():
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.info("Adding '{}' foreground for {}".format(name,tag.upper()))
                    self.log.debug("Adding '{}' foreground for {}".format(name,tag.upper()))
                    kwargs = dict(lmax=self.lmax_win, freqs=self.frequencies, mode=tag.upper(), survey=self.survey)
                    if isinstance(self.foregrounds[tag.upper()][name], str):
                        kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds[tag.upper()][name])
                    self.fgs[tag].append(fg_list[name](**kwargs))

        if self.use_tt: self.log.debug(f"nbintt: {self.nbintt}")
        if self.use_te: self.log.debug(f"nbinte: {self.nbinte}")
        if self.use_ee: self.log.debug(f"nbinee: {self.nbinee}")

        self.log.info(f"Init ACTpol_{self.survey} likelihood done")


    def _dl2cl( self, dl):
        cl2dl = np.ones( self.lmax_win+1)
        
        lth = np.arange( self.lmax_win+1)
        cl2dl[1:] = (lth*(lth+1)/2./np.pi)[1:]

        return dl/cl2dl


    def loglike(self, dl_cmb, **params):
        """
        dl_cmb: Dl TT
        """

        yp1 = params['yp1']
        yp2 = params['yp2']
        ct1 = params['cal98']  #Cal and leakage errors included in covmat and so fixed to 1
        ct2 = params['cal150'] #Cal and leakage errors included in covmat and so fixed to 1
        a1  = params['leak98']
        a2  = params['leak150']

        #Calculate CMB+fg
        dlth = {
            'tt':np.repeat([dl_cmb['tt'][:self.lmax_win+1]],self.nspectt,axis=0),
            'te':np.repeat([dl_cmb['te'][:self.lmax_win+1]],self.nspecte,axis=0),
            'ee':np.repeat([dl_cmb['ee'][:self.lmax_win+1]],self.nspecee,axis=0)
            }

        #zero l>6000
        for k in dlth:
            dlth[k][:,6000:] = 0.
#        print( "CMB: ", dlth['tt'][:,340:360])

        #FORCE MODEL bf_ACTPol_lcdm.minimum.theory_cl
        ell, dltt, dlte, dlee, _, _ = np.loadtxt( "/sps/planck/Users/tristram/Soft/Hillik/modules/data/actpolfull_dr4.01/data/bf_ACTPol_lcdm.minimum.theory_cl", unpack=True)
        dlth['tt'][:,np.array(ell,int)] = dltt
        dlth['ee'][:,np.array(ell,int)] = dlee
        dlth['te'][:,np.array(ell,int)] = dlte
        #WARNING
        
        for tag in dlth.keys():
            for fg in self.fgs[tag]:
#                print( f"{fg.name}: ", fg.compute_dl( params)[:,1000:1010])
                dlth[tag] += fg.compute_dl( params) #array( nspecf, lmax_win+1)

        #Get theory in Cls
        X_theory = dlth
        for k,val in X_theory.items():
            val[:,:2] = 0.
            X_theory[k] = self._dl2cl(val)
#        print( "Total: ", X_theory['ee'][0,2:10])
#        print( "Total: ", X_theory['ee'][0,self.lmax_win-10:])

#        print( np.shape(X_theory['tt']))
#        print( np.shape( self.win_func))

        # Get binned model
        X_model = []
        for s in range(self.nspectt):
            ishift = s*self.nbintt
            X_model += list( self.win_func[ishift:ishift+self.nbintt] @ X_theory['tt'][s] )
        for s in range(self.nspecte):
            ishift = self.nspectt*self.nbintt + s*self.nbinte
            X_model += list( self.win_func[ishift:ishift+self.nbinte] @ X_theory['te'][s] )
        for s in range(self.nspecee):
            ishift = self.nspectt*self.nbintt + self.nspecte*self.nbinte + s*self.nbinee
            X_model += list( self.win_func[ishift:ishift+self.nbinee] @ X_theory['ee'][s] )
        X_model = np.asarray(X_model)

        # Add leakage
        ntt = self.nbintt
        nte = self.nbinte
        nee = self.nbinee
        X_model[3*ntt+0*nte:3*ntt+1*nte] += X_model[0*ntt:1*ntt]*a1*self.l98[:nte]
        X_model[3*ntt+1*nte:3*ntt+2*nte] += X_model[1*ntt:2*ntt]*a2*self.l150[:nte]
        
        X_model[3*ntt+2*nte:3*ntt+3*nte] += X_model[1*ntt:2*ntt]*a1*self.l98[:nte]
        X_model[3*ntt+3*nte:3*ntt+4*nte] += X_model[2*ntt:3*ntt]*a2*self.l150[:nte]

        X_model[3*ntt+4*nte+0*nee:3*ntt+4*nte+1*nee] += 2*X_model[3*ntt+0*nte:3*ntt+1*nte]*a1*self.l98[:nte] + X_model[0*ntt:1*ntt]*(a1*self.l98[:nte])**2.
        X_model[3*ntt+4*nte+1*nee:3*ntt+4*nte+2*nee] += ( X_model[3*ntt+1*nte:3*ntt+2*nte]*a1*self.l98[:nte] +
                                                          X_model[3*ntt+2*nte:3*ntt+3*nte]*a2*self.l150[:nte] +
                                                          X_model[1*ntt:2*ntt]*a1*self.l98[:nte]*a2*self.l150[:nte])
        X_model[3*ntt+4*nte+2*nee:3*ntt+4*nte+3*nee] += 2*X_model[3*ntt+3*nte:3*ntt+4*nte]*a2*self.l150[:nte] + X_model[2*ntt:3*ntt]*(a2*self.l150[:nte])**2.

        # Calibrate
        X_model[0*ntt:1*ntt] *= ct1*ct1
        X_model[1*ntt:2*ntt] *= ct1*ct2
        X_model[2*ntt:3*ntt] *= ct2*ct2
        X_model[3*ntt+0*nte:3*ntt+1*nte] *= ct1*ct1*yp1
        X_model[3*ntt+1*nte:3*ntt+2*nte] *= ct1*ct2*yp2
        X_model[3*ntt+2*nte:3*ntt+3*nte] *= ct1*ct2*yp1
        X_model[3*ntt+3*nte:3*ntt+4*nte] *= ct1*ct2*yp2
        X_model[3*ntt+4*nte+0*nee:3*ntt+4*nte+1*nee] *= ct1*ct1*yp1*yp1 
        X_model[4*nte+3*ntt+1*nee:3*ntt+4*nte+2*nee] *= ct1*ct2*yp1*yp2
        X_model[3*ntt+4*nte+2*nee:3*ntt+4*nte+3*nee] *= ct2*ct2*yp2*yp2

        # Select data
        bstart = 0
        bend   = self.nbint
        if self.use_tt and not self.use_te and not self.use_ee:
            bstart = 0
            bend   = self.nbintt*self.nspectt
        if not self.use_tt and self.use_te and not self.use_ee:
            bstart = self.nbintt*self.nspectt
            bend   = self.nbintt*self.nspectt + self.nbinte*self.nspecte
        if not self.use_tt and not self.use_te and self.use_ee:
            bstart = self.nbintt*self.nspectt + self.nbinte*self.nspecte
            bend   = self.nbint

#        print( "Data: ", self.b_dat[bstart:bend])
#        print( "Model: ", X_model[bstart:bend])

        diff_vec = (self.b_dat - X_model)[bstart:bend]
        fisher   = self.covmat[bstart:bend,bstart:bend]

#        print( "Res: ", diff_vec**2)
#        print( "Var: ", np.diag(fisher))
#        print( "chi2: ", diff_vec**2/np.diag(fisher))

        # Invert covmat
        fisher = np.linalg.inv( fisher)

#        print( "chi2: ", diff_vec**2*np.diag(fisher))

        #chi2
        dlnlike = np.sum( diff_vec @ fisher @ diff_vec)
        
        self.log.debug(f"lnlike = {dlnlike} / {len(diff_vec)}")
        
        return -0.5*dlnlike

    def get_requirements(self):
        requirements = dict(Cl={mode: self.lmax_win+1 for mode in ["tt","te","ee"]})
        return requirements

    def logp(self, **params_values):
        dl = self.theory.get_Cl(units="muK2", ell_factor=True)
        return self.loglike(dl, **params_values)


class TT(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 TT dataset
    """

class EE(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 EE dataset
    """

class TE(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 TE dataset
    """

class wide(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 full dataset
    """

class deep(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 full dataset
    """
