""".. module:: SPT-3G

:Synopsis: Definition of python-native CMB likelihood for SPT-3G TT+TE+EE. Adapted from Fortran
likelihood code https://pole.uchicago.edu/public/data/balkenhol22/SPT3G_2018_TTTEEE_public_likelihood.zip
[Balkenhold et al. 2022]

:Author: Matthieu Tristram

"""

import itertools
import os
import re
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import packages_path_input
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

default_spectra_list = [
    "90_Tx90_T",
    "90_Tx90_E",
    "90_Ex90_E",
    "90_Tx150_T",
    "90_Tx150_E",
    "90_Ex150_E",
    "90_Tx220_T",
    "90_Tx220_E",
    "90_Ex220_E",
    "150_Tx150_T",
    "150_Tx150_E",
    "150_Ex150_E",
    "150_Tx220_T",
    "150_Tx220_E",
    "150_Ex220_E",
    "220_Tx220_T",
    "220_Tx220_E",
    "220_Ex220_E",
]


class SPT3GPrototype(InstallableLikelihood):
    install_options = {
        "download_url": "https://pole.uchicago.edu/public/data/balkenhol22/SPT3G_2018_TTTEEE_public_likelihood.zip",
        "data_path": "spt3g_2018",
    }

    bibtex_file = "spt3g_2022.bibtex"

    bin_min: Optional[int] = 1
    bin_max: Optional[int] = 44
    windows_lmin: Optional[int] = 1
    windows_lmax: Optional[int] = 3200

    spectra_to_fit: Optional[Sequence[str]] = default_spectra_list
    foregrounds: Optional[dict]

    # fmt: off
    spec_bin_min: Optional[Sequence[int]] = [10,  1,  1, 10,  1,  1, 10,  1,  1, 10,  1,  1, 15,  1,  1, 15,  1,  1]
    spec_bin_max: Optional[Sequence[int]] = [44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44]

    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "spt3g_2018/SPT3G_2018_TTTEEE_public_likelihood/data/SPT3G_2018_TTTEEE"
    # fmt: on
    bandpower_filename: Optional[str]
    covariance_filename: Optional[str]
    fiducial_covariance_filename: Optional[str]
    beam_covariance_filename: Optional[str]
    cal_covariance_filename: Optional[str]
    window_folder: Optional[str]

    cov_eval_cut_threshold: Optional[float] = 0.2
    cov_eval_large_number_replacement: Optional[float] = 1e3
    beam_cov_scale: Optional[float] = 1.0
    aberration_coefficient: Optional[float] = -0.0004826

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to SPT3G data. Set the likelihood property 'path' or "
                f"the common property '{packages_path_input}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError( self.log, f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].",)
        self.fgds_folder = os.path.join(data_file_path, self.fgds_folder)
        if not os.path.exists(self.fgds_folder):
            raise LoggedError( self.log, f"The 'fgds_folder' directory does not exist. Check the given path [{self.fgds_folder}].")

        #define the survey
        self.survey = "SPT3G"

        # Get likelihood name and add the associated mode
        lkl_name = self.__class__.__name__.upper()
        self.use_cl = [lkl_name[i : i + 2] for i in range(0, len(lkl_name), 2)]

        for b, spec in enumerate(self.spectra_to_fit):
            if self.spec_bin_min[b] <= 0 or self.spec_bin_max[b] >= 45:
                raise LoggedError(
                    self.log, f"SPT-3G 2018 TTTEEE: bad ell range selection for spectrum: {spec}"
                )

        # Check if a late crop is requested and read in the mask if necessary
        # MT: Not Implemented

        # Compute cross-spectra frequencies and mode given the spectra name to fit
        r = re.compile("(.+?)_(.)x(.+?)_(.)")
        self.cross_frequencies = [r.search(spec).group(1, 3) for spec in self.spectra_to_fit]
        self.cross_spectra = ["".join(r.search(spec).group(2, 4)) for spec in self.spectra_to_fit]
        self.frequencies = sorted(
            {int(freq) for freqs in self.cross_frequencies for freq in freqs}
        )
        self.log.debug(f"Using cross-frequencies {self.cross_frequencies}")
        self.log.debug(f"Using cross-spectra {self.cross_spectra}")
        self.log.debug(f"Using {self.frequencies} GHz frequency bands")

        # Determine how many spectra are TT vs TE vs EE and the total number of bins we are fitting
        self.N_s_TT = np.sum([c == "TT" for c in self.cross_spectra])
        self.N_s_TE = np.sum([c == "TE" for c in self.cross_spectra])
        self.N_s_EE = np.sum([c == "EE" for c in self.cross_spectra])

        # Determine how many different frequencies get used
        self.N_freq = len(self.frequencies)

        # Band Powers
        self.bandpowers = np.loadtxt(
            os.path.join(self.data_folder, self.bandpower_filename), unpack=True
        )
        self.bandpowers = self.bandpowers.reshape(-1, self.bin_max)

        #-----------------------------------------------
        # Covariance Matrix
        #-----------------------------------------------
        bp_cov = np.loadtxt(os.path.join(self.data_folder, self.covariance_filename))
        fid_cov = np.loadtxt(os.path.join(self.data_folder, self.covariance_filename))
        #        fid_cov = np.loadtxt(os.path.join(self.data_folder, self.fiducial_covariance_filename))

        #-----------------------------------------------
        # Beam Covariance Matrix
        #-----------------------------------------------
        self.beam_cov = np.loadtxt(os.path.join(self.data_folder, self.beam_covariance_filename))
        self.beam_cov = self.beam_cov * self.beam_cov_scale

        #-----------------------------------------------
        # Windows Functions
        #-----------------------------------------------
        # These are a bit trickier to handle due to the independent cuts possible for TT/TE/EE
        # The windows for low ell TT spectra exist in the files so that we can read these in in a nice array
        # Re-order/crop later when the binning is performed
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_folder, f"window_{i}.txt"),
                    unpack=True,
                )[1:]
                for i in range(self.bin_min, self.bin_max + 1)
            ]
        )

        #-----------------------------------------------
        # Compute spectra/cov indices given spectra to fit
        #-----------------------------------------------
        vec_indices = np.array([default_spectra_list.index(spec) for spec in self.spectra_to_fit])
        cov_indices = np.concatenate(
            [
                np.arange(
                    i * self.bin_max + self.spec_bin_min[i] - 1,
                    i * self.bin_max + self.spec_bin_max[i],
                    dtype=int,
                )
                for i in vec_indices
            ]
        )
        self.bandpowers = self.bandpowers[vec_indices]
        self.windows = self.windows[:, vec_indices, :]
        self.spec_bin_min = np.array(self.spec_bin_min)[vec_indices]
        self.spec_bin_max = np.array(self.spec_bin_max)[vec_indices]
        self.N_b_total = np.sum(self.spec_bin_max - self.spec_bin_min + 1)  # total nb of bins
        self.N_s = len(vec_indices)  # nb of spectra
        if len(cov_indices) != self.N_b_total:
            raise LoggedError(
                self.log,
                f"Total number of bin is not consistent {len(cov_indices)} (expected {self.N_b_total})",
            )
        for i in range(self.N_s):
            self.log.debug(
                "\t {:>3}x{:>3}-{}: [{:2d}-{:2d}]".format(
                    *self.cross_frequencies[i],
                    self.cross_spectra[i],
                    self.spec_bin_min[i],
                    self.spec_bin_max[i],
                )
            )

        #-----------------------------------------------
        # Select spectra/cov elements given indices
        #-----------------------------------------------
        self.log.debug(f"Selected bp ({self.N_s}): {vec_indices}")
#        self.log.debug(f"Selected cov indices ({self.N_b_total}): {cov_indices}")
        self.bp_cov   = bp_cov[np.ix_(cov_indices, cov_indices)]
        self.fid_cov  = fid_cov[np.ix_(cov_indices, cov_indices)]
        self.beam_cov = self.beam_cov[np.ix_(cov_indices, cov_indices)]

        # Ensure covariance is positive definite
        self._bp_cov_posdef = self.bp_cov

        #-----------------------------------------------
        # Calibration Covariance
        #-----------------------------------------------
        # The order of the cal covariance is T90, T150, T220, E90, E150, E220
        calib_cov = np.loadtxt(os.path.join(self.data_folder, self.cal_covariance_filename))
        cal_indices = np.array([[90.0, 150.0, 220.0].index(freq) for freq in self.frequencies])
        if "TE" in self.cross_spectra:
            cal_indices = np.concatenate([cal_indices, cal_indices + 3])
        elif "TT" not in self.cross_spectra:
            # Only polar calibrations shift by 3
            cal_indices = cal_indices + 3
        calib_cov = calib_cov[np.ix_(cal_indices, cal_indices)]
        self.inv_calib_cov = np.linalg.inv(calib_cov)
        self.calib_params = np.array(
            ["SPT3G_cal_{}{}".format(*p) for p in itertools.product(["T", "E"], [90, 150, 220])]
        )[cal_indices]
        self.log.debug(f"Calibration parameters: {self.calib_params}")

        self.lmin = self.windows_lmin
        self.lmax = self.windows_lmax

        #-----------------------------------------------
        # Initialise foreground model
        #-----------------------------------------------
        self.fgs = {"TT":[],"TE":[],"EE":[]}
        for tag in self.fgs.keys():
            if tag in self.cross_spectra:
                for name in self.foregrounds[tag.upper()].keys():
                    if name not in fg_list.keys():
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.debug("Adding '{}' foreground for {}".format(name,tag))
                    kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode=tag, auto=True, survey=self.survey)
                    if isinstance(self.foregrounds[tag.upper()][name], str):
                        kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds[tag.upper()][name])
                    elif name == "szxcib":
                        filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["tsz"])
                        filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["cib"])
                        kwargs["filenames"] = (filename_tsz,filename_cib)
                    self.fgs[tag].append(fg_list[name](**kwargs))

        self.log.info(f"SPT-3G 2022: Likelihood successfully initialised!")


    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl.lower(): self.lmax for cl in self.use_cl}}


    def compute_chi2(self, dl_cmb, **params):

        ells = np.arange(self.lmin, self.lmax+1)

        dlfg = {}
        for mode in self.use_cl:
            dlfg[mode] = np.zeros((sum([c == mode for c in self.cross_spectra]),self.lmax+1))
            for fg in self.fgs[mode]: dlfg[mode] += fg.compute_dl( params)

        db_model = np.empty_like(self.bandpowers)
        for i, (cross_spectrum, cross_frequency) in enumerate(
            zip(self.cross_spectra, self.cross_frequencies)
        ):

            # Add CMB
            dl_model = dl_cmb[cross_spectrum][ells]

            # Add super sample lensing
            dl_model += self.ApplySuperSampleLensing(params.get("kappa"), dl_model)

            # Add Aberration correction
            dl_model += self.ApplyAberrationCorrection(self.aberration_coefficient, dl_model)
            
            # Add foregrounds
            dl_model += dlfg[cross_spectrum][fg._cross_frequencies.index(tuple(map(int,cross_frequency)))][ells]
            
            # Apply calibration
            cal = params.get("SPT3G_cal") * self.ApplyCalibration(
                params.get(f"SPT3G_cal_{cross_spectrum[0]}{cross_frequency[0]}"),
                params.get(f"SPT3G_cal_{cross_spectrum[1]}{cross_frequency[1]}"),
                params.get(f"SPT3G_cal_{cross_spectrum[0]}{cross_frequency[1]}"),
                params.get(f"SPT3G_cal_{cross_spectrum[1]}{cross_frequency[0]}")
            )
            dl_model = dl_model / cal
            
            # Binning via window and concatenate
            db_model[i] = self.windows[:, i, :] @ dl_model

        # Select bins and calculate difference of theory and data
        self.log.debug("Compute residuals")
        delta_data_model = np.concatenate(
            [
                (self.bandpowers[i] - db_model[i])[self.spec_bin_min[i] - 1 : self.spec_bin_max[i]]
                for i in range(self.N_s)
            ]
        )
        dbs = np.concatenate(
            [db_model[i][self.spec_bin_min[i] - 1 : self.spec_bin_max[i]] for i in range(self.N_s)]
        )

        # Add the beam coariance to the band power covariance
        self.log.debug("Add beam cov")
        cov_for_logl = self._bp_cov_posdef + self.beam_cov * np.outer(dbs, dbs)
        
        # Final crop to ignore select band powers
        # MT: not implemented
        
        # Compute chisq
        self.log.debug("Compute chi2")
        chi2, slogdet = self._gaussian_loglike(cov_for_logl, delta_data_model, cholesky=True)

        self.log.debug(f"SPT3G chi2/ndof = {chi2:.14f}/{len(delta_data_model)}")
        return chi2, slogdet

    def loglike(self, dl_cmb, **params):

        chi2, slogdet = self.compute_chi2( dl_cmb, **params)

        # Apply calibration prior
        self.log.debug("Apply calibration prior")
        delta_cal = np.array([params.get(p) - 1.0 for p in self.calib_params])
        cal_prior = delta_cal @ self.inv_calib_cov @ delta_cal

        self.log.debug(f"SPT3G detcov = {slogdet:.14f}")
        self.log.debug(f"SPT3G cal. prior = {cal_prior:.14f}")
        return -0.5 * (chi2 + slogdet + cal_prior)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(
            {"TT": Cls.get("tt"), "TE": Cls.get("te"), "EE": Cls.get("ee")}, **data_params
        )

    def dof( self):
        return len(self._bp_cov_posdef)
    
    def _gaussian_loglike(self, dlcov, res, cholesky=True):
        """
        Returns -Log Likelihood for Gaussian: (d^T Cov^{-1} d + log|Cov|)
        """

        if cholesky:
            from scipy.linalg import cho_factor, cho_solve

            L, low = cho_factor(dlcov)

            # compute ln det
            slogdet = 2.0 * np.sum(np.log(np.diag(L)))

            # Compute C-1.d
            invCd = cho_solve((L, low), res)

            # Compute chi2
            chi2 = res @ invCd

        else:
            chi2 = res @ np.linalg.inv(dlcov) @ res
            sign, slogdet = np.linalg.slogdet(dlcov)

        return chi2, slogdet

    # Super sample lensing
    # Based on Manzotti et al. 2014 (https://arxiv.org/pdf/1401.7992.pdf) Eq. 32
    # Applies correction to the spectrum and returns the correction slotted into the fg array
    def ApplySuperSampleLensing(self, kappa, Dl_theory):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.lmax + 1)

        # Grab Cl derivative
        Cl_derivative = self._GetClDerivative(Dl_theory)

        # Calculate super sample lensing correction
        # (In Cl space) SSL = -k/l^2 d/dln(l) (l^2Cl) = -k(l*dCl/dl + 2Cl)
        ssl_correction = ells * Cl_derivative  # l*dCl/dl
        ssl_correction = (
            ssl_correction * ells * (ells + 1) / (2 * np.pi)
        )  # Convert this part to Dl space already
        ssl_correction = ssl_correction + 2 * Dl_theory  # 2Cl - but already converted to Dl
        ssl_correction = -kappa * ssl_correction  # -kappa

        return ssl_correction

    # Aberration Correction
    # Based on Jeong et al. 2013 (https://arxiv.org/pdf/1309.2285.pdf) Eq. 23
    # Applies correction to the spectrum and returns the correction by itself
    def ApplyAberrationCorrection(self, ab_coeff, Dl_theory):
        # AC = beta*l(l+1)dCl/dln(l)/(2pi)

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.lmax + 1)

        # Grab Cl derivative
        Cl_derivative = self._GetClDerivative(Dl_theory)

        # Calculate aberration correction
        # (In Cl space) AC = -coeff*dCl/dln(l) = -coeff*l*dCl/dl
        # where coeff contains the boost amplitude and direction (beta*<cos(theta)> in Jeong+ 13)
        aberration_correction = -ab_coeff * Cl_derivative * ells
        aberration_correction = (
            aberration_correction * ells * (ells + 1) / (2 * np.pi)
        )  # Convert to Dl

        return aberration_correction

    # Helper to get the derivative of the spectrum
    # Takes Dl in, but returns Cl derivative#
    # Handles end points approximately
    # Smoothes any spike at the ell_max
    def _GetClDerivative(self, Dl_theory):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.lmax + 1)

        # Calculate derivative
        Cl_derivative = Dl_theory * 2 * np.pi / (ells * (ells + 1))  # Convert to Cl
        Cl_derivative[1:-1] = 0.5 * (Cl_derivative[2:] - Cl_derivative[:-2])  # Find gradient
        Cl_derivative[0] = Cl_derivative[1]  # Handle start approximately
        Cl_derivative[-1] = Cl_derivative[-2]  # Handle end approximately

        # Smooth over spike at lmax
        # Transition point between Boltzmann solver Cl and where the spectrum comes from a lookup table/interpolation can cause a spike in derivative
        #  if (CosmoSettings%lmax_computed_cl .LT. lmax-1) then
        #    Cl_derivative(CosmoSettings%lmax_computed_cl) = 0.75*Cl_derivative(CosmoSettings%lmax_computed_cl-1) + 0.25*Cl_derivative(CosmoSettings%lmax_computed_cl+2)
        #    Cl_derivative(CosmoSettings%lmax_computed_cl+1) = 0.75*Cl_derivative(CosmoSettings%lmax_computed_cl+2) + 0.25*Cl_derivative(CosmoSettings%lmax_computed_cl-1)

        return Cl_derivative

    # Calibration
    # Data is scaled as: TT: T1*T2, TE: 0.5*(T1*E2+T2*E1), EE: E1*E2
    # Theory is scaled by the inverse
    # In function this is calculated as  0.5*(cal1*cal2+cal3*cal4)
    def ApplyCalibration(self, cal1, cal2, cal3, cal4):

        # This is how the data spectra are calibrated
        calibration = 0.5 * (cal1 * cal2 + cal3 * cal4)
        return calibration



class TTTEEE(SPT3GPrototype):
    r"""
    Likelihood for Balkenhold et al. 2022
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, TT/TE/EE
    Written by Lennart Balkenhol
    """


class TT(SPT3GPrototype):
    r"""
    Likelihood for Balkenhold et al. 2022
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, TT
    Written by Lennart Balkenhol
    """


class TE(SPT3GPrototype):
    r"""
    Likelihood for Balkenhold et al. 2022
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, TE
    Written by Lennart Balkenhol
    """


class EE(SPT3GPrototype):
    r"""
    Likelihood for Balkenhold et al. 2022
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, EE
    Written by Lennart Balkenhol
    """
