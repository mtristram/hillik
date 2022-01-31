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

fg_list = 
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

#    frequencies: Sequence[int] = [98, 150]

    data_folder: Optional[str] = "actpol_full_dr4/likelihood"
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
    use_tt: Optional[bool] = True  #TT only
    use_te: Optional[bool] = True  #TE only
    use_ee: Optional[bool] = True  #EE only

    bool use_deep = True
    bool use_wide = True

    #--------------------------------------------------------------
    #Settings (should not be altered)
    #--------------------------------------------------------------
    # general settings
    #--------------------------------------------------------------
    int tt_lmax = 6000

    #----------------------------------------------------------------
    # likelihood terms from ACT data
    #----------------------------------------------------------------
    int nnu      = 2   # number of frequencies
    int nspectot = 10  #nspecf*nspec+2 TE for 90x150 and 150x90
    int nspecf   = 3   #95x95, 95x150, 150
    int nspectt  = 3   #TT
    int nspecte  = 4   #TE
    int nspecee  = 3   #EE
    int nbintt = 52    #max nbins in ACT TT data  
    int nbinte = 52    #max nbins in ACT TE data
    int nbinee = 52    #max nbins in ACT EE data
    int nbint  = 520   #total bins
    int lmax_win = 7925 #ell max of the full window functions 
    int bmax0  = 52     #number of bins in full window function
    int b0=5           # setting bins discarded in TT (i.e., ell>600)
    int b1=0           # bins discarded for TE
    int b2=0           # bins discarded for EE

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
        self.data_folder = os.path.join(self.data_folder, "data/actpol_full_dr4.01/data")

        #-----------------------------------------------
        #load spectrum
        #-----------------------------------------------

        # read bandpowers (1040)
        # check file before
        self.b_dat = np.loadtxt(os.path.join(self.data_folder, self.spec_filename), unpack=True)
        self.b_dat = self.b_dat[0:nbint]

        # Read windows (520,7924) check l ?
        # check file before
        self.win_func = np.loadtxt(os.path.join(self.data_folder, self.bbl_filename))
        nbint,lmax_win = np.shape(self.win_func)
        if nbint != self.nbint or lmax_win != self.lmax_win:
            raise LoggedError(self.log, "Wrong size for Window function (%s)"%self.bbl_filename)

        # Read leakage (52)
        # check file before
        ell, self.l98, self.l150 = np.loadtxt(os.path.join(self.data_folder, self.leakd_filename),unpack=True)
        if len(ell) != self.nbinte:
            raise LoggedError(self.log, "Wrong size for Leakage template (%s)"%self.leakd_filename)

        #-----------------------------------------------
        #Read Covariance Matrix
        #-----------------------------------------------
        #covmat (1040,1040)
        self.covmat = np.loadtxt( os.path.join(self.data_folder, self.cov_filename))
        self.covmat = self.covmat[:nbint,:nbint]

        #cut lmin TT
        for s in range(self.nspectt):
            ishift = s*self.nbintt
            covmat[ishift:ishift+b0,ishift:ishift+b0] = identity(b0)

        #cut lmin TE
        for s in range(self.nspecte):
            ishift = self.nspectt*self.nbintt + s*self.nbinte
            covmat[ishift:ishift+b1,ishift:ishift+b1] = identity(b1)

        #cut lmin EE
        for s in range(self.nspecee):
            ishift = self.nspectt*self.nbintt + self.nspecte*self.nbinte + s*self.nbinee
            covmat[ishift:ishift+b2,ishift:ishift+b2] = identity(b2)

        # Init foreground model
        self.fgs = {'tt':[],'te':[],'ee':[]}
        for tag in fgs.keys():
            for name in self.foregrounds[tag.upper()].keys():
                if name not in fg_list.keys():
                    raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                self.log.debug("Adding '{}' foreground for {}".format(name,tag.upper()))
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode=tag.upper())
                if isinstance(self.foregrounds[tag.upper()][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds[tag.upper()][name])
                self.fgsTT[tag].append(fg_list[name](**kwargs)))

        self.log.debug(f"nbintt: {self.nbintt}")
        self.log.debug(f"nbinte: {self.nbinte}")
        self.log.debug(f"nbinee: {self.nbinee}")

        self.log.info("Init ACTpol_deep likelihood done")


    def loglike(self, dl_cmb, **params):
        """
        dl_cmb: Dl TT
        """

        ct1 = 1.0  #Cal and leakage errors included in covmat and so fixed here
        ct2 = 1.0
        a1  = 1.0
        a2  = 1.0

        #Calculate CMB+fg
        dlth = {'tt':np.repeat([dl_cmb['tt']],self.nspectt),
                'te':np.repeat([dl_cmb['te']],self.nspecte),
                'ee':np.repeat([dl_cmb['ee']],self.nspecee)}
        for tag in dlth.keys():
            for fg in self.fgs[tag]:
                dlth_tt += fg.compute_dl( params, mode=tag) #array( nspecf, lmax_win)

        #Get theory in Cls
        lth = np.arange(self.lmax_win)
        X_theory = dlth
        for k,val in dlth.items():
            val /= (lth*(lth+1)/2/np.pi)

        # Get binned model
        X_model = []
        for s in range(self.nspectt):
            ishift = s*self.nbintt
            X_model += list( self.win[ishift:ishift+self.nbintt] @ X_theory['tt'][s] )
        for s in range(self.nspecte):
            ishift = self.nspectt * self.nbintt + s* self.nbinte
            X_model += list( self.win[ishift:ishift+*self.nbinte] @ X_theory['te'][s] )
        for s in range(self.nspecee):
            ishift = self.nspectt * self.nbintt + self.nspecte * self.nbinte + s* self.nbinee
            X_model += list( self.win[ishift:ishift+*self.nbinee] @ X_theory['ee'][s] )
        X_model = np.asarray(X_model)

        # Add leakage
        X_model[3*nbintt+0*nbinte:3*nbintt+1*nbinte] += X_model[0*nbintt:1*nbintt]*a1*l98[:nbinte]
        X_model[3*nbintt+1*nbinte:3*nbintt+2*nbinte] += X_model[1*nbintt:2*nbintt]*a2*l150[:nbinte]
        
        X_model[3*nbintt+2*nbinte:3*nbintt+3*nbinte] += X_model[1*nbintt:2*nbintt]*a1*l98[:nbinte]
        X_model[3*nbintt+3*nbinte:3*nbintt+4*nbinte] += X_model[2*nbintt:3*nbintt]*a2*l150[:nbinte]

        X_model[3*nbintt+4*nbinte+0*nbinee:3*nbintt+4*nbinte+1*nbinee] += 2*X_model[3*nbintt+0*nbinte:3*nbintt+1*nbinte]*a1*l98[:nbinte] + X_model[0*nbintt:1*nbintt]*(a1*l98[:nbinte])**2.
        X_model[3*nbintt+4*nbinte+1*nbinee:3*nbintt+4*nbinte+2*nbinee] += ( X_model[3*nbintt+1*nbinte:3*nbintt+2*nbinte]*a1*l98[:nbinte] +
                                                                            X_model[3*nbintt+2*nbinte:3*nbintt+3*nbinte]*a2*l150[:nbinte] +
                                                                            X_model[1*nbintt:2*nbintt]*a1*l98[:nbinte]*a2*l150[:nbinte]
                                                                            )
        X_model[3*nbintt+4*nbinte+2*nbinee:3*nbintt+4*nbinte+3*nbinee] += 2*X_model[3*nbintt+3*nbinte:3*nbintt+4*nbinte]*a2*l150[:nbinte] + X_model[2*nbintt:3*nbintt]*(a2*l150[:nbinte])**2.

        # Calibrate
        yp1 = params['yp1']
        yp2 = params['yp2']
        X_model[0*nbintt:1*nbintt] *= ct1*ct1
        X_model[1*nbintt:2*nbintt] *= ct1*ct2
        X_model[2*nbintt:3*nbintt] *= ct2*ct2
        X_model[3*nbintt:3*nbintt+nbinte] *= ct1*ct1*yp1
        X_model[3*nbintt+1*nbinte:3*nbintt+2*nbinte] *= ct1*ct2*yp2
        X_model[3*nbintt+2*nbinte:3*nbintt+3*nbinte] *= ct1*ct2*yp1
        X_model[3*nbintt+3*nbinte:3*nbintt+4*nbinte] *= ct1*ct2*yp2
        X_model[3*nbintt+4*nbinte+0*nbinee:3*nbintt+4*nbinte+1*nbinee] *= ct1*ct1*yp1*yp1 
        X_model[4*nbinte+3*nbintt+1*nbinee:3*nbintt+4*nbinte+2*nbinee] *= ct1*ct2*yp1*yp2
        X_model[3*nbintt+4*nbinte+2*nbinee:3*nbintt+4*nbinte+3*nbinee] *= ct2*ct2*yp2*yp2

        # Select data
        bstart = 0
        bend   = nbint
        if use_tt and not use_te and not use_ee:
            bstart = 0
            bend   = nbintt*nspectt
        if not use_tt and use_te and not use_ee:
            bstart = nbintt*nspectt
            bend   = nbintt*nspectt + nbinte*nspecte
        if not use_tt and not use_te and use_ee:
            bstart = nbintt*nspectt + nbinte*nspecte
            bend   = nbint

        diff_vec = (self.b_dat - Xmodel)[bstart:bend]
        fisher   = self.covmat[bstart:bend,bstart:bend]

        # Invert covmat
        fisher = np.linalg.inv( fisher)
        
        #chi2
        dlnlike = sum( diff_vec @ fisher @ diff_vec)

        self.log.debug(f"SPTHiEllLnLike lnlike = {SPTHiEllLnLike} (with priors)")
        self.log.debug(f"Calibration chisq = {2 * CalibLnLike}")
        self.log.debug(f"lnLcov term = {detcov}")
        self.log.debug(f"chisq for cov only: {2 * LnL}")
        self.log.debug(f"chisq for FG prior: {2 * FGPriorLnLike}")
        self.log.debug(f"SPTHiEllLnLike chisq (after prior) = {2 * (SPTHiEllLnLike - detcov)}")

        return -0.5*dlnlike

    def get_requirements(self):
        requirements = dict(Cl={mode: self.lmax for mode in ["tt","te","ee"]})
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


class full(ACTPolLikelihood):
    """
    CMB likelihood with ACTpol DR4 full dataset
    """
