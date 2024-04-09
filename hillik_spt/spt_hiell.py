import os
from typing import Optional, Sequence

import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

import hillik_foregrounds as fg

fg_list = {
    "cib": fg.cib,
    "radio_poisson": fg.ps_radio,
    "cib_poisson": fg.ps_dusty,
    "poisson": fg.ps,
    "dust": fg.dust,
    "tsz": fg.tsz,
    "ksz": fg.ksz,
    "szxcib": fg.szxcib,
    }


class SPTHiellLikelihood(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/SPT/reichardt_2020/likelihood.tar.gz",
        "data_path": "spt_hiell_2020",
    }

    cal_cov = [
        [1.1105131e-05, 3.5551351e-06, 1.1602891e-06],
        [3.5551351e-06, 3.4153547e-06, 2.1348018e-06],
        [1.1602891e-06, 2.1348018e-06, 1.7536000e-05],
    ]

    frequencies: Sequence[int] = [95, 150, 220]
    ReportFGLmax = 13500
    BoltzmannLmax: Optional[int] = 10000
    
    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "spt_hiell_2020/likelihood"
    desc_file: Optional[str]
    bp_file: Optional[str]
    cov_file: Optional[str]
    beamerr_file: Optional[str]
    window_file: Optional[str]
    bin0: Optional[int] = 0 #cut first bins
    
    normalizeSZ_143GHz: Optional[bool] = True
    callFGprior: Optional[bool] = True
    applyFTSprior: Optional[bool] = True
    
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

        #define the survey
        self.survey = "SPT"

        # Init foreground model
        self.fgs = []
        for name in self.foregrounds["TT"].keys():
            if name not in fg_list.keys():
                raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

            self.log.debug("Adding '{}' foreground".format(name))
            kwargs = dict(lmax=self.ReportFGLmax, freqs=self.frequencies, mode='TT', auto=True, survey=self.survey, emulator=False)
            if isinstance(self.foregrounds["TT"][name], str):
                kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds["TT"][name])
                if not os.path.exists(kwargs["filename"]):
                    kwargs["emulator"] = True
            elif name == "szxcib":
                filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["tsz"])
                filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["cib"])
                kwargs["filenames"] = (filename_tsz,filename_cib)
            self.fgs.append(fg_list[name](**kwargs))

        # Update data_folder location
        self.data_folder = os.path.join(self.data_folder, "data/spt_hiell_2020")

        # get info from the desc_file
        self._update_with_desc_file()

        self.log.debug(f"nall: {self.nall}")
        self.log.debug(f"nfreq: {self.nfreq}")
        self.log.debug(f"spt_windows_lmin: {self.spt_windows_lmin}")
        self.log.debug(f"spt_windows_lmax: {self.spt_windows_lmax}")
        self.log.debug(f"Boltzmann lmax: {self.BoltzmannLmax}")

        self.lmin = self.spt_windows_lmin
        self.lmax = self.spt_windows_lmax

        if self.spt_windows_lmax > self.ReportFGLmax:
            raise LoggedError(self.log, "Hard-wired lmax in foregrounds is too low for SPT_hiell")

        if self.spt_windows_lmin < 2 or self.spt_windows_lmin >= self.spt_windows_lmax:
            raise LoggedError(self.log, "Invalid lranges for sptpol")

        # read bandpowers (90x90, 90x150, 90x220, 150x150, 150x220, 220x220)
        # check file before
        bla, self.spec = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)

        # read covariance
        # check file before
        self.cov = np.fromfile(
            os.path.join(self.data_folder, self.cov_file), dtype=np.float64
        ).reshape(self.nall, self.nall)

        # read beam_err
        # check file before
        self.beam_err = np.fromfile(
            os.path.join(self.data_folder, self.beamerr_file), dtype=np.float64
        ).reshape(self.nall, self.nall)

        # Read windows
        # check file before
        self.windows = self._read_windows(os.path.join(self.data_folder, self.window_file))

        # define indicies
        self.indices = []
        for j in range(self.nfreq):
            for k in range(j, self.nfreq):
                self.indices.append((j, k))

        # define offsets for xfreq in Cl vector
        self.offsets = [0]
        for i in range(1, self.nband):
            self.offsets.append(self.offsets[i - 1] + self.nbins[i - 1])

        self.log.info("Initialized!")

    def _read_windows(self, filename):
        import struct

        with open(filename, "rb") as f:
            efflmin, efflmax = struct.unpack("@II", f.read(2 * np.dtype(np.int32).itemsize))

        if efflmax < self.spt_windows_lmin or efflmin > self.spt_windows_lmax:
            raise LoggedError(self.log, "unallowed l-ranges for binary window functions")

        offset = 2 * (np.dtype(np.int32).itemsize)
        windows = np.fromfile(filename, dtype=np.float64, offset=offset).reshape(self.nall, -1)

        return windows

    def _update_with_desc_file(self):
        filename = os.path.join(self.data_folder, self.desc_file)
        with open(filename) as f:
            self.nall, self.nfreq = [int(float(x)) for x in next(f).split()]
            self.nband = int(self.nfreq * (self.nfreq + 1) / 2)

            self.nbins = [int(next(f)) for i in range(self.nband)]
            if self.nall != sum(self.nbins):
                raise LoggedError(self.log, "Mismatched number of bandpowers")

            self.spt_norm_fr = [float(next(f)) for i in range(5)]

            if self.normalizeSZ_143GHz:
                self.log.debug("Using 143 as tSZ center freq")
                self.spt_norm_fr[4] = 143.0

            self.spt_windows_lmin, self.spt_windows_lmax = [int(x) for x in next(f).split()]

            eff_fr = []
            for _ in range(self.nfreq):
                eff_fr.append([float(next(f)) for i in range(5)])
            self.spt_eff_fr = np.array(eff_fr)

            self.spt_prefactor = np.array([float(next(f)) for i in range(self.nfreq)])

    def _gaussian_loglike(self, dlcov, res):
        """
        Returns -Log Likelihood for Gaussian: (d^T Cov^{-1} d + log|Cov|)/2
        """
        from scipy.linalg import cho_factor, cho_solve

        L, low = cho_factor(dlcov)

        # compute det
        detcov = 2.0 * np.sum(np.log(np.diag(L)))

        # Compute C-1.d
        invCd = cho_solve((L, low), res)

        # Compute chi2
        chi2 = res @ invCd

        return chi2, detcov

    def compute_chi2(self, dl_boltz, **params):
        """
        dl_cmb: Dl TT
        """
        Cal = params[f"{self.survey}_cal"]
        CalFactors = [params[f"{self.survey}_cal_{nu}"] for nu in self.frequencies]

        dl_cmb = np.zeros( self.lmax+1)
        dl_cmb[:self.BoltzmannLmax] = dl_boltz['tt'][:self.BoltzmannLmax]

        dl_fg = np.zeros( (self.nband, self.lmax+1) )
#        dlfg = []
        for fg in self.fgs:
            dl_fg += fg.compute_dl( params)
#            dlfg.append( fg.compute_dl(params))
#        print( "write fgs templates")
#        np.save( "hillik_spt_fgs", np.array(dlfg))

        # Loop on nband
        cbs = []
        cbd = []
        for i in range(self.nband):
            j, k = self.indices[i]
            thisoffset = self.offsets[i]
            thisnbin = self.nbins[i]

            # get theory spectra
            dl_th = dl_cmb[self.lmin : self.lmax+1] + dl_fg[i, self.lmin : self.lmax+1]

            # bin theory with window functions
            tmpcb = self.windows[thisoffset+self.bin0:thisoffset+thisnbin] @ dl_th

            # apply prefactors
            tmpcb = tmpcb * self.spt_prefactor[k] * self.spt_prefactor[j] * CalFactors[j] * CalFactors[k] * Cal

            cbs += list(tmpcb)
            cbd += list(self.spec[thisoffset+self.bin0:thisoffset+thisnbin])
            
        # Residuals
        self.delta_cl = np.array(cbs) - np.array(cbd)

        # Dl covariance (with beams)
        indices = []
        for i in range(self.nband):
            indices += list(np.arange(self.offsets[i]+self.bin0,self.offsets[i]+self.nbins[i]))
        cov_w_beam = self.cov[np.ix_(indices,indices)] + self.beam_err[np.ix_(indices,indices)] * np.outer(cbs, cbs)

        # compute LogLike
        LnL, detcov = self._gaussian_loglike(cov_w_beam, self.delta_cl)

        self.log.debug(f"chisq for cov only: {LnL} / {len(self.delta_cl)}")
        return LnL, detcov


    def loglike(self, dl_boltz, **params):
        CalFactors = [params[f"{self.survey}_cal_{nu}"] for nu in self.frequencies]
        FTSfactor = params["FTS_calibration_error"]

        chi2, detcov = self.compute_chi2( dl_boltz, **params)
        SPTHiEllLnLike = chi2 + detcov

        # Add FG priors
#        if self.callFGprior:
#            FGPriorLnLike = self.fg.getForegroundPriorLnL(params)
#            SPTHiEllLnLike += FGPriorLnLike

        # Add calib LogLike
        delta_calib = np.log(CalFactors)
        CalibLnLike, _ = self._gaussian_loglike(self.cal_cov, delta_calib)
        SPTHiEllLnLike += CalibLnLike

        # Add FTS prior
        # Prior is 0.3 GHz for 1 sigma around 0.
        if self.applyFTSprior:
            FTSLnLike = (FTSfactor / 0.3) ** 2
            SPTHiEllLnLike += FTSLnLike

#        self.log.debug(f"SPTHiEllLnLike lnlike = {SPTHiEllLnLike} (with priors)")
        self.log.debug(f"Calibration chisq = {CalibLnLike}")
#        self.log.debug(f"lnLcov term = {detcov}")
        self.log.debug(f"LnL (after priors) = {SPTHiEllLnLike}")

        return -0.5 * SPTHiEllLnLike

    def get_requirements(self):
        requirements = dict(Cl={mode: self.BoltzmannLmax for mode in ["tt"]})
        return requirements

    def logp(self, **params_values):
        dl = self.provider.get_Cl(units="muK2", ell_factor=True)
        return self.loglike(dl, **params_values)

    def dof(self):
        indices = []
        for i in range(self.nband):
            indices += list(np.arange(self.offsets[i]+self.bin0,self.offsets[i]+self.nbins[i]))
        return len(indices)

    def reduction_matrix(self):
        X = np.zeros( (sum(self.nbins), 15) )

        i=0
        for nb in self.nbins:
            for ib in range(nb):
                X[i,ib] = 1.
                i += 1

        return X


class TThighl(SPTHiellLikelihood):
    """
    CMB likelihood with SPT-SZ and SPTpol surveys (Reichard et al. 2020)
    """
