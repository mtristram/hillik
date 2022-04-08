import os
from typing import Optional, Sequence

import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

import hillik_foregrounds as fg

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

    data_folder: Optional[str]
    spec_filename: Optional[str]
    cov_filename: Optional[str]
    bbl_filename: Optional[str]
    leakd_filename: Optional[str]

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
        #Read Covariance Matrix
        #-----------------------------------------------
        #covmat (1040,1040)
        self.covmat = np.loadtxt( os.path.join(self.data_folder, self.cov_filename))
        self.covmat = self.covmat[:self.nbint,:self.nbint]

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
        self._is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE"]}
        self.log.debug("mode = {}".format(self._is_mode))

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
        self.fgs = {'TT':[],'TE':[],'EE':[]}
        for tag,is_used in self._is_mode.items():
            if is_used:
                for name in self.foregrounds[tag.upper()].keys():
                    if name not in fg_list.keys():
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.info("Adding '{}' foreground for {}".format(name,tag.upper()))
                    kwargs = dict(lmax=self.lmax_win, freqs=self.frequencies, mode=tag.upper(), survey=self.survey)
                    if isinstance(self.foregrounds[tag.upper()][name], str):
                        kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds[tag.upper()][name])
                    self.fgs[tag].append(fg_list[name](**kwargs))

        if self._is_mode['TT']: self.log.debug(f"nbintt: {self.nbintt}")
        if self._is_mode['TE']: self.log.debug(f"nbinte: {self.nbinte}")
        if self._is_mode['EE']: self.log.debug(f"nbinee: {self.nbinee}")

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

        cal = params['A_actpol']
        yp1 = params['yp1']
        yp2 = params['yp2']
        ct1 = params['cal98']  #Cal and leakage errors included in covmat and so fixed to 1
        ct2 = params['cal150'] #Cal and leakage errors included in covmat and so fixed to 1
        a1  = params['leak98']
        a2  = params['leak150']

        #Calculate CMB+fg
        dlth = { 'tt':np.zeros( (self.nspectt,self.lmax_win+1) ),
                 'te':np.zeros( (self.nspecte,self.lmax_win+1) ),
                 'ee':np.zeros( (self.nspecee,self.lmax_win+1) )}
        for tag in ['tt','te','ee']:
            dlth[tag][:,:self.tt_lmax+1] = dl_cmb[tag][:self.tt_lmax+1]

        #TEST FORCE MODEL bf_ACTPol_lcdm.minimum.theory_cl
#        ell, dltt, dlte, dlee, _, _ = np.loadtxt( "/sps/planck/Users/tristram/Soft/Hillik/modules/data/actpolfull_dr4.01/data/bf_ACTPol_lcdm.minimum.theory_cl", unpack=True)
#        dlth['tt'][:,np.array(ell,int)] = dltt
#        dlth['ee'][:,np.array(ell,int)] = dlee
#        dlth['te'][:,np.array(ell,int)] = dlte
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
        X_model[3*ntt+4*nte+0*nee:3*ntt+4*nte+1*nee] += 2*X_model[3*ntt+0*nte:3*ntt+1*nte]*a1*self.l98[:nte] + X_model[0*ntt:1*ntt]*(a1*self.l98[:nte])**2.
        #EE(98x150) <- TE(98x150) + TE(150x98) + TT(98x150)
        X_model[3*ntt+4*nte+1*nee:3*ntt+4*nte+2*nee] += ( X_model[3*ntt+1*nte:3*ntt+2*nte]*a1*self.l98[:nte] +
                                                          X_model[3*ntt+2*nte:3*ntt+3*nte]*a2*self.l150[:nte] +
                                                          X_model[1*ntt:2*ntt]*a1*self.l98[:nte]*a2*self.l150[:nte])
        #EE(150x150) <- 2*TE(150x150) + TT(150x150)
        X_model[3*ntt+4*nte+2*nee:3*ntt+4*nte+3*nee] += 2*X_model[3*ntt+3*nte:3*ntt+4*nte]*a2*self.l150[:nte] + X_model[2*ntt:3*ntt]*(a2*self.l150[:nte])**2.
        
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
        bstart = 0
        bend   = self.nbint
        if self._is_mode['TT'] and not _is_mode['TE'] and not _is_mode['EE']:
            bstart = 0
            bend   = self.nbintt*self.nspectt
        if not self._is_mode['TT'] and self._is_mode['TE'] and not self._is_mode['EE']:
            bstart = self.nbintt*self.nspectt
            bend   = self.nbintt*self.nspectt + self.nbinte*self.nspecte
        if not self._is_mode['TT'] and not self._is_mode['TE'] and self._is_mode['EE']:
            bstart = self.nbintt*self.nspectt + self.nbinte*self.nspecte
            bend   = self.nbint

        diff_vec = (self.b_dat - X_model)[bstart:bend]
        fisher   = self.covmat[bstart:bend,bstart:bend]

        # Invert covmat
        fisher = np.linalg.inv( fisher)

        #chi2
        dlnlike = np.sum( diff_vec @ fisher @ diff_vec)
        
        self.log.debug(f"lnlike = {dlnlike} / {len(diff_vec)}")
        
        return -0.5*dlnlike

    def get_requirements(self):
        requirements = dict(Cl={mode:self.tt_lmax for mode in ["tt","te","ee"]})
        return requirements

    def logp(self, **params_values):
        dl = self.theory.get_Cl(units="muK2", ell_factor=True)
        return self.loglike(dl, **params_values)

