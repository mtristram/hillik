""".. module:: ACTpol_full_DR4

:Synopsis: Definition of python-native CMB likelihood for ACT likelihood.
Adapted from Fortran likelihood code
https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.cfm
full ACTPol spectra at 98x98, 98x150 and 150x150 GHz from 350 < l < 8000 measured during 2013-2016 in temperature and polarization

:Author: Matthieu Tristram

"""
import os
from typing import Optional, Sequence

import hillik_foregrounds as fg
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

fg_list = {
    "cib": fg.cib,
    "poisson": fg.ps,
    "galactic_dust": fg.dust,
    "tsz": fg.tsz,
    "ksz": fg.ksz,
    "szxcib": fg.szxcib,
    }



class ACTPolLikelihood(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr4/likelihoods/actpolfull_dr4.01.tar.gz",
        "data_path": "actpol_full_dr4",
    }

    frequencies: Sequence[int] = [98, 150]

    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "actpol_full_dr4/actpolfull_dr4.01/data"
    spec_filename: Optional[str]
    cov_filename: Optional[str]
    bbl_filename: Optional[str]
    leakd_filename: Optional[str]

    #--------------------------------------------------------------
    #Settings (should not be altered)
    #--------------------------------------------------------------
    # general settings
    #--------------------------------------------------------------
    BoltzmannLmax = 6000

    #----------------------------------------------------------------
    # likelihood terms from ACT data
    #----------------------------------------------------------------
    nnu      = 2     # number of frequencies
    nspectot = 10    #TT90x90, TT90x150, TT150x150, TE90x90, TE90x150, TE150x90, TE150x150, EE90x90, EE90x150, EE150x150
    nspecf   = 3     #95x95, 95x150, 150x150
    nspectt  = 3     #TT
    nspecte  = 4     #TE
    nspecee  = 3     #EE
    nbintt = 52      #max nbins in ACT TT data
    nbinte = 52      #max nbins in ACT TE data
    nbinee = 52      #max nbins in ACT EE data
    nbint  = 520     #total bins
    lmax_win = 7925  #ell max of the full window functions
    bmax0  = 52      #number of bins in full window function
    bminTT=5             # setting bins discarded in TT (i.e., ell>600)
    bminTE=0             # bins discarded for TE
    bminEE=0             # bins discarded for EE
#    bminTT=33            # setting bins discarded in TT (i.e., ell>2000)
#    bminTE=23            # bins discarded for TE (i.e. ell>1500)
#    bminEE=13            # bins discarded for EE (i.e. ell>1000)

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
            raise LoggedError( self.log, f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].")
        self.fgds_folder = os.path.join(data_file_path, self.fgds_folder)
        if not os.path.exists(self.fgds_folder):
            raise LoggedError( self.log, f"The 'fgds_folder' directory does not exist. Check the given path [{self.fgds_folder}].")

        # Update data_folder location
#        self.data_folder = os.path.join(self.data_folder, "data/actpol_full_dr4.01/data")

        #define the survey
        self.survey = ""
        if self.use_wide and self.use_deep:
            raise LoggedError(self.log, "Choose survey DEEP or WIDE, not both")
        if self.use_wide: self.survey = "ACTw"
        if self.use_deep: self.survey = "ACTd"
        self.log.debug( f"Survey = {self.survey}")

        #check modes
        # Get likelihood name and add the associated mode
        likelihood_name = self.__class__.__name__
        likelihood_modes = [likelihood_name[i:i+2] for i in range(0,len(likelihood_name),2)]
        self._is_mode = {mode.lower(): mode in likelihood_modes for mode in ["TT", "TE", "EE"]}
        self.log.debug("mode = {}".format(self._is_mode))

        #-----------------------------------------------
        #load spectrum
        #-----------------------------------------------

        # read bandpowers (1040)
        # check file before
        self.b_dat = np.loadtxt(os.path.join(self.data_folder, self.spec_filename), unpack=True)
        self.b_dat = self.b_dat[0:self.nbint]

        # Read windows (520,7924)
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
        #Select Data
        #-----------------------------------------------
        self._bstart = 0
        self._bend   = self.nbint
        if self._is_mode['tt'] and not self._is_mode['te'] and not self._is_mode['ee']:
            self._bstart = 0
            self._bend   = self.nbintt*self.nspectt
        if not self._is_mode['tt'] and self._is_mode['te'] and not self._is_mode['ee']:
            self._bstart = self.nbintt*self.nspectt
            self._bend   = self.nbintt*self.nspectt + self.nbinte*self.nspecte
        if not self._is_mode['tt'] and not self._is_mode['te'] and self._is_mode['ee']:
            self._bstart = self.nbintt*self.nspectt + self.nbinte*self.nspecte
            self._bend   = self.nbint

        #-----------------------------------------------
        #Read Covariance Matrix
        #-----------------------------------------------
        
        #covmat (1040,1040)
        covmat = np.loadtxt( os.path.join(self.data_folder, self.cov_filename))
        covmat = covmat[:self.nbint,:self.nbint]

        #cut lmin TT
        for s in range(self.nspectt):
            ishift = s*self.nbintt
            covmat[ishift:ishift+self.bminTT,ishift:ishift+self.bminTT] = np.identity(self.bminTT)*1e10

        #cut lmin TE
        for s in range(self.nspecte):
            ishift = self.nspectt*self.nbintt + s*self.nbinte
            covmat[ishift:ishift+self.bminTE,ishift:ishift+self.bminTE] = np.identity(self.bminTE)*1e10

        #cut lmin EE
        for s in range(self.nspecee):
            ishift = self.nspectt*self.nbintt + self.nspecte*self.nbinte + s*self.nbinee
            covmat[ishift:ishift+self.bminEE,ishift:ishift+self.bminEE] = np.identity(self.bminEE)*1e10

        #invert covmat
        covmat = covmat[self._bstart:self._bend,self._bstart:self._bend]
        self.fisher = np.linalg.inv( covmat)

        # Init foreground model
        self.fgs = {'tt':[],'te':[],'ee':[]}
        for tag,is_used in self._is_mode.items():
            if is_used:
                for name in self.foregrounds[tag.upper()].keys():
                    if name not in fg_list.keys():
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.debug("Adding '{}' foreground for {}".format(name,tag.upper()))
                    kwargs = dict(lmax=self.lmax_win, freqs=self.frequencies, mode=tag.upper(), auto=True, survey=self.survey)
                    if isinstance(self.foregrounds[tag.upper()][name], str):
                        kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds[tag.upper()][name])
                    elif name == "szxcib":
                        filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["tsz"])
                        filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["cib"])
                        kwargs["filenames"] = (filename_tsz,filename_cib)
                    self.fgs[tag].append(fg_list[name](**kwargs))

        if self._is_mode['tt']: self.log.debug(f"nbintt: {self.nbintt}")
        if self._is_mode['te']: self.log.debug(f"nbinte: {self.nbinte}")
        if self._is_mode['ee']: self.log.debug(f"nbinee: {self.nbinee}")

        self.log.info("Initialized!")


    def _dl2cl( self, dl):
        cl2dl = np.ones( self.lmax_win+1)

        lth = np.arange( self.lmax_win+1)
        cl2dl[1:] = (lth*(lth+1)/2./np.pi)[1:]

        return dl/cl2dl


    def compute_chi2(self, dl_cmb, **params):
        """
        dl_cmb: Dl TT
        """
        surv = self.survey[:-1]
        cal = params[f'cal_{surv}']
        ct1 = params[f'cal_{surv}_98']  #Cal and leakage errors included in covmat and so fixed to 1
        ct2 = params[f'cal_{surv}_150'] #Cal and leakage errors included in covmat and so fixed to 1
        yp1 = params[f'poleff_{surv}_98']
        yp2 = params[f'poleff_{surv}_150']
        a1  = params[f'leak_{surv}_98']
        a2  = params[f'leak_{surv}_150']

        #Calculate CMB+fg
        dlth = { 'tt':np.zeros( (self.nspectt,self.lmax_win+1) ),
                 'te':np.zeros( (self.nspecte,self.lmax_win+1) ),
                 'ee':np.zeros( (self.nspecee,self.lmax_win+1) )}
        for tag in ['tt','te','ee']:
            dlth[tag][:,:self.BoltzmannLmax+1] = dl_cmb[tag][:self.BoltzmannLmax+1]

        for tag in dlth.keys():
#            dlfg = []
            for fg in self.fgs[tag]:
                dlth[tag] += fg.compute_dl( params) #array( nspecf, lmax_win+1)
#                dlfg.append( fg.compute_dl(params))
#            print( "write fgs templates")
#            np.save( f"hillik_{self.survey}_fgs_{tag}", np.array(dlfg))

        #Get theory in Cls
        X_theory = dlth
        for k,val in X_theory.items():
            val[:,:2] = 0.
            X_theory[k] = self._dl2cl(val)

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

        # Add leakage (Warning: need same binning in tt, te and ee)
        # TiEj = TiEj + TiTj*gamma_j
        # EiEj = EiEj + TiEj*gamma_i + TjEi*gamma_j + TiTj*gamma_i*gamma_j
        ntt = self.nbintt
        nte = self.nbinte
        nee = self.nbinee
        X_model[3*ntt+0*nte:3*ntt+1*nte] += X_model[0*ntt:1*ntt]*a1*self.l98[:nte]    #TE(98x98)   <- TT(98x98)
        X_model[3*ntt+1*nte:3*ntt+2*nte] += X_model[1*ntt:2*ntt]*a2*self.l150[:nte]   #TE(98x150)  <- TT(98x150)
        X_model[3*ntt+2*nte:3*ntt+3*nte] += X_model[1*ntt:2*ntt]*a1*self.l98[:nte]    #TE(150x98)  <- TT(98x150)
        X_model[3*ntt+3*nte:3*ntt+4*nte] += X_model[2*ntt:3*ntt]*a2*self.l150[:nte]   #TE(150x150) <- TT(150x150)

        #EE(98x98) <- 2*TE(98x98) + TT(98x98)
        X_model[3*ntt+4*nte+0*nee:3*ntt+4*nte+1*nee] += ( 2*X_model[3*ntt+0*nte:3*ntt+1*nte]*a1*self.l98[:nte]
                                                          + X_model[0*ntt:1*ntt]*(a1*self.l98[:nte])**2.)
        #EE(98x150) <- TE(98x150) + TE(150x98) + TT(98x150)
        X_model[3*ntt+4*nte+1*nee:3*ntt+4*nte+2*nee] += ( X_model[3*ntt+1*nte:3*ntt+2*nte]*a1*self.l98[:nte] +
                                                          X_model[3*ntt+2*nte:3*ntt+3*nte]*a2*self.l150[:nte] +
                                                          X_model[1*ntt:2*ntt]*a1*self.l98[:nte]*a2*self.l150[:nte])
        #EE(150x150) <- 2*TE(150x150) + TT(150x150)
        X_model[3*ntt+4*nte+2*nee:3*ntt+4*nte+3*nee] += ( 2*X_model[3*ntt+3*nte:3*ntt+4*nte]*a2*self.l150[:nte]
                                                          + X_model[2*ntt:3*ntt]*(a2*self.l150[:nte])**2.)

        # Calibrate
        X_model[0*ntt:1*ntt] *= cal*ct1*ct1
        X_model[1*ntt:2*ntt] *= cal*ct1*ct2
        X_model[2*ntt:3*ntt] *= cal*ct2*ct2
        X_model[3*ntt+0*nte:3*ntt+1*nte] *= cal*ct1*ct1*yp1
        X_model[3*ntt+1*nte:3*ntt+2*nte] *= cal*ct1*ct2*yp2
        X_model[3*ntt+2*nte:3*ntt+3*nte] *= cal*ct1*ct2*yp1
        X_model[3*ntt+3*nte:3*ntt+4*nte] *= cal*ct1*ct2*yp2
        X_model[3*ntt+4*nte+0*nee:3*ntt+4*nte+1*nee] *= cal*ct1*ct1*yp1*yp1
        X_model[4*nte+3*ntt+1*nee:3*ntt+4*nte+2*nee] *= cal*ct1*ct2*yp1*yp2
        X_model[3*ntt+4*nte+2*nee:3*ntt+4*nte+3*nee] *= cal*ct2*ct2*yp2*yp2

        # Select data
        self.delta_cl = (self.b_dat - X_model)[self._bstart:self._bend]

        #chi2
        lnlike = self.delta_cl @ self.fisher @ self.delta_cl

        self.log.debug(f"chisq = {lnlike} / {sum(np.diag(self.fisher>1e-9))}")

        return lnlike

    def get_requirements(self):
        requirements = dict(Cl={mode:self.BoltzmannLmax for mode in ["tt","te","ee"]})
        return requirements

    def logp(self, **params_values):
        dl = self.theory.get_Cl(units="muK2", ell_factor=True)
        return self.loglike(dl, **params_values)

    def loglike(self, dl_cmb, **params):
        return -0.5*self.compute_chi2( dl_cmb, **params)

    def dof( self):
        return sum(np.diag(self.fisher>1e-9))
        
    def reduction_matrix( self, mode='tt'):
        if mode == 'tt':
            nbin,nspec = self.nbintt, self.nspectt
        elif mode == 'te':
            nbin,nspec = self.nbinte, self.nspecte
        elif mode == 'ee':
            nbin,nspec = self.nbinee, self.nspecee

        X = np.zeros( (len(self.delta_cl), nbin) )
        
        for ix in range(nspec):
            for ib in range(nbin):
                X[ix*nbin+ib,ib] = 1.

        return X
