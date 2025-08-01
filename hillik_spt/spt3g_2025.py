""".. module:: SPT-3G

:Synopsis: Definition of python-native CMB likelihood for SPT-3G TT+TE+EE [Camphuis et al. 2025]
Adapted from previous python version of the SPT likelihood codes (https://github.com/xgarrido/spt_likelihoods)
and CANDL likelihood code (https://github.com/Lbalkenhol/candl)
Data are for SPT3G-D1 (https://github.com/SouthPoleTelescope/spt_candl_data)

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

default_spectra_list = [
    "TT 90x90",
    "TE 90x90",
    "EE 90x90",
    "TT 90x150",
    "TE 90x150",
    "TE 150x90",
    "EE 90x150",
    "TT 90x220",
    "TE 90x220",
    "TE 220x90",
    "EE 90x220",
    "TT 150x150",
    "TE 150x150",
    "EE 150x150",
    "TT 150x220",
    "TE 150x220",
    "TE 220x150",
    "EE 150x220",
    "TT 220x220",
    "TE 220x220",
    "EE 220x220",
]

default_bin_min = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
default_bin_max = [52, 72, 72, 52, 72, 72, 72, 52, 72, 72, 72, 52, 72, 72, 52, 72, 72, 72, 52, 72, 72]


#from "effective_frequencies.yaml"
feff = {
    "tsz":   {90:95.6933, 150:148.849, 220:220.15},
    "dust":  {90:95.9631, 150:150.012, 220:222.773},
    "cib":   {90:95.96, 150:150.00, 220:222.76},
    "radio": {90:94.40, 150:146.00, 220:212.7}, #SPT3G_2018_TTTEEE_effective_band_centres.dat
    "sync":  {90:94.40, 150:146.00, 220:212.7}, #SPT3G_2018_TTTEEE_effective_band_centres.dat
}

class SPT3G_D1_Lik(InstallableLikelihood):
    install_options = {
        "github_repository": "SouthPoleTelescope/spt_candl_data",
#        "directory": "spt_candl_data-main/spt_candl_data/SPT3G_D1_TnE_v0",
#        "data_path": "spt3g_2025/SPT3G_D1_TnE_v0",
        }
    type = "CMB"

    bibtex_file = "spt3g_2025.bibtex"

    spectra_to_fit: Optional[Sequence[str]] = default_spectra_list
    foregrounds: Optional[dict]

    # fmt: off
    spec_bin_min: Optional[Sequence[int]] = default_bin_min
    spec_bin_max: Optional[Sequence[int]] = default_bin_max

    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "spt_candl_data/spt_candl_data/SPT3G_D1_TnE_v0"
    # fmt: on

    bandpower_filename: Optional[str] = "SPT3G_D1_TnE_bdp.txt"
    covariance_filename: Optional[str] = "SPT3G_D1_TnE_cov.dat"
    beam_eigenmodes_filename: Optional[str] = "beams_templates/cov_eigenmodes_300_4100.npz"
    beam_main_filename: Optional[str] = "beams_templates/B_ell_300_4100_main_rc4.npz"
    window_folder: Optional[str] = "windows/"

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

        # Check if a late crop is requested and read in the mask if necessary
        # MT: Not Implemented

        # Compute cross-spectra frequencies and mode given the spectra name to fit
        r = re.compile("([A-Z]{2})\s(\d+)x(\d+)")
        self.cross_frequencies = [r.search(spec).group(2, 3) for spec in self.spectra_to_fit]
        self.cross_spectra = ["".join(r.search(spec).group(1)) for spec in self.spectra_to_fit]
        self.frequencies = sorted(
            {int(freq) for freqs in self.cross_frequencies for freq in freqs}
        )
#        self.log.debug(f"Using cross-frequencies {self.cross_frequencies}")
#        self.log.debug(f"Using cross-spectra {self.cross_spectra}")
        self.log.debug(f"Using {self.frequencies} GHz frequency bands")

        # Determine how many different frequencies get used
        self.N_freq = len(self.frequencies)

        #-----------------------------------------------
        # Band Powers
        #-----------------------------------------------
        data = np.loadtxt( os.path.join(self.data_folder, self.bandpower_filename))
        nbins_per_spec = np.array(default_bin_max) - np.array(default_bin_min) + 1
        ishift = [0]+list(np.cumsum(nbins_per_spec))
        self.bandpowers = {
            spec:data[ishift[i]:ishift[i+1]] for i,spec in enumerate(default_spectra_list)
            }

        #-----------------------------------------------
        # Covariance Matrix
        #-----------------------------------------------
        bp_cov  = np.loadtxt(os.path.join(self.data_folder, self.covariance_filename))

        #-----------------------------------------------
        # Windows Functions
        #-----------------------------------------------
        # These are a bit trickier to handle due to the independent cuts possible for TT/TE/EE
        # The windows for low ell TT spectra exist in the files so that we can read these in in a nice array
        # Re-order/crop later when the binning is performed
        ells = []
        self.windows = {}
        for spec in self.spectra_to_fit:
            data = np.loadtxt(
                os.path.join(self.data_folder, self.window_folder, f"{spec.replace(' ','_')}_window_functions.txt")
                )
            ells = ells+list(data.T[0])
            self.windows[spec] = data.T[1:]
        self.lmin = int(min(ells))
        self.lmax = int(max(ells))
        self.log.debug( f"l-range: {self.lmin}, {self.lmax}")

        #-----------------------------------------------
        # Compute cov indices given spectra to fit
        #-----------------------------------------------
        vec_indices = np.array([default_spectra_list.index(spec) for spec in self.spectra_to_fit])
        cov_indices = np.concatenate(
            [
                np.arange(
                    ishift[i] + self.spec_bin_min[i] - 1,
                    ishift[i] + self.spec_bin_max[i],
                    dtype=int,
                )
                for i in vec_indices
            ]
        )
        self.log.debug(f"Selected {len(vec_indices)} spectra: {[spec for spec in self.spectra_to_fit]}")

        nbins_total = np.sum([bin_max-bin_min+1 for bin_min,bin_max in zip(self.spec_bin_min,self.spec_bin_max)])  # total nb of bins
        if len(cov_indices) != nbins_total:
            raise LoggedError(
                self.log,
                f"Total number of bin is not consistent {len(cov_indices)} (expected {nbins_total})",
            )
        for i,spec in enumerate(self.spectra_to_fit):
            self.log.debug(
                "\t {}: [{:2d}-{:2d}]".format(
                    spec, 
                    self.spec_bin_min[i],
                    self.spec_bin_max[i],
                )
            )

        self._inv_bpcov = np.linalg.inv(bp_cov[np.ix_(cov_indices, cov_indices)])


        #-----------------------------------------------
        # Beams
        #-----------------------------------------------
        # Temperature Main beam for all multipoles normalized at l=800
        data = np.load(os.path.join(self.data_folder, self.beam_main_filename))
        self.beam_main_temperature = {fq:data[fq] for fq in ['90','150','220']}

        # Beam eigenmodes ordered as freq (90,150,220) and given for all multipoles
        data = np.load(os.path.join(self.data_folder, self.beam_eigenmodes_filename))
        self.beam_eigenmodes = {}
        for ifq,freq in enumerate(['90','150','220']):
            self.beam_eigenmodes[freq] = data['modes'][ifq*len(data['ell']):(ifq+1)*len(data['ell'])].T
        del(data)


        #-----------------------------------------------
        # Initialise foreground model
        #-----------------------------------------------
        self.fgs = {"TT":[],"TE":[],"EE":[]}
        for tag in self.fgs.keys():
            if tag in self.cross_spectra:
                cross = [tuple(int(x) for x in xfq) for xfq,cs in zip(self.cross_frequencies,self.cross_spectra) if cs == tag]
                for name in self.foregrounds[tag.upper()].keys():
                    if not hasattr(fg,name):
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.debug("Adding '{}' foreground for {}".format(name,tag))
                    kwargs = dict(lmax=self.lmax, cross=cross, mode=tag, survey=self.survey, feff=feff)
                    if 'dust' in name: kwargs['lnorm'] = 80
                    if isinstance(self.foregrounds[tag.upper()][name], str):
                        kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds[tag.upper()][name])
                    elif "szxcib" in name:
                        key_with_tsz = next((k for k in self.foregrounds["TT"] if 'tsz' in k), None)
                        kwargs["filename_tsz"] = self.foregrounds["TT"][key_with_tsz] and os.path.join(self.fgds_folder, self.foregrounds["TT"][key_with_tsz])
                        key_with_cib = next((k for k in self.foregrounds["TT"] if 'cib' in k and k != "szxcib"), None)
                        kwargs["filename_cib"] = self.foregrounds["TT"][key_with_cib] and os.path.join(self.fgds_folder, self.foregrounds["TT"][key_with_cib])
                    self.fgs[tag].append(getattr(fg,name)(**kwargs))

        self.log.info(f"SPT-3G 2025: Likelihood successfully initialised!")


    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl.lower(): self.lmax for cl in self.use_cl}}


    def _get_spec_info( self, spec):
        pattern = re.compile("([A-Z]{2})\s(\d+)x(\d+)")
        cross_spectrum, freq1, freq2 = pattern.search(spec).groups()

        return cross_spectrum, (freq1,freq2)

    def compute_sky_model( self, dl_cmb, **params):
        """
        Compute the angular power spectrum of the sky model

        Result on Dl [muK2]
        """

        ells = np.arange(self.lmin, self.lmax+1)

        # Foregrounds (for all multipoles)
        dlfg = {}
        for mode in self.use_cl:
            dlfg[mode] = np.zeros((sum([c == mode for c in self.cross_spectra]),self.lmax+1))
            for fg in self.fgs[mode]: dlfg[mode] += fg.compute_dl( params)

        # Sky Model
        sky_model = {}
        for spec in self.spectra_to_fit:
            cross_spectrum, cross_frequency = self._get_spec_info(spec)

            # Add CMB
            dl_model = dl_cmb[cross_spectrum][ells]

            # Add super sample lensing
            dl_model += self.SuperSampleLensing(params.get("SPT3G_kappa"), dl_model)

            # Add Aberration correction
            dl_model += self.AberrationCorrection(self.aberration_coefficient, dl_model)
            
            # Add foregrounds
            cross_list = [xfq for xfq,cs in zip(self.cross_frequencies,self.cross_spectra) if cs == cross_spectrum]
            dl_model += dlfg[cross_spectrum][cross_list.index(cross_frequency)][ells]

            sky_model[spec] = dl_model

        return sky_model


    def apply_spt_corrections( self, sky_model, **params):

#        dl_model = sky_model.copy()
        dl_model = {}
        for spec in self.spectra_to_fit:
            cross_spectrum, cross_frequency = self._get_spec_info(spec)
            dl_model[spec] = np.copy(sky_model[spec])
            
            # T2P leakage
            sigmas = {'90':0.000274,'150':0.000192,'220':0.000169}
            dls = {tag:sky_model[tag+' '+"x".join(sorted(cross_frequency,key=int))] for tag in ['TT','TE','EE']}
            dls['ET'] = sky_model['TE'+' '+"x".join(sorted(cross_frequency,key=int)[::-1])]
            dl_model[spec] += self.T2PLeakage( [sigmas[fq] for fq in cross_frequency],
                                               [params.get(f"SPT3G_T2P2_{fq}") for fq in cross_frequency],
                                               dls)[cross_spectrum]
            
            # Beam eigenmodes
#            dl_model[spec] *= self.BeamEigenModes( cross_frequency,
#                                                   [params.get(f"SPT3G_beta_{n+1}") for n in range(9)])
            
            # Beam polar
#            dl_model[spec] *= self.PolarizedBeam( cross_spectrum,
#                                                  cross_frequency,
#                                                  [params.get(f"SPT3G_beta_pol_{fq}") for fq in cross_frequency])

            #Beam corrections
            dl_model[spec] *= self.ModesAndPolarizedBeam( cross_spectrum,
                                                          cross_frequency,
                                                          [params.get(f"SPT3G_beta_pol_{fq}") for fq in cross_frequency],
                                                          [params.get(f"SPT3G_beta_{n+1}") for n in range(9)])
            
            # Apply calibration
            dl_model[spec] /= (
                params.get(f"SPT3G_cal") *
                self.InternalCalibration( cross_spectrum,
                                          [params.get(f"SPT3G_cal_{fq}") for fq in cross_frequency],
                                          [params.get(f"SPT3G_pe_{fq}") for fq in cross_frequency]
                                          )
                )
        
        return dl_model


    def compute_chi2(self, dl_cmb, **params):
        """
        From Eq. (40) in Camphuis et al 2025 (https://arxiv.org/abs/2506.20707)

        Cl^model = Acal . Ecal . Q . B_P(beta_pol) . B_T(beta_i) . L(epsilon) . [ A . S(kappa) . Cl^CMB + Cl^fg ]

        with corrections:
            Acal: calibration
            Ecal: polarisation efficiency
            Q: Bandpower window functions
            B_P: polarized beam correction (beta_pol for each freq)
            B_T: beam error modes (one beta for each of the 9 eigenmodes)
            L: quadrupolar beam leakage
            A: Aberration
            S: super-sample lensing correction (kappa)
        """

        # Compute sky model (fg + Aberration + SuperSampleLensing)
        sky_model = self.compute_sky_model( dl_cmb, **params)

        # Apply SPT intrumental effects
        dl_model = self.apply_spt_corrections( sky_model, **params)

        # Binning via window and concatenate
        db_model = {spec:self.windows[spec] @ dl_model[spec] for spec in self.spectra_to_fit}

        # Select bins and calculate difference of theory and data
        print( "Compute residuals")
        self.log.debug("Compute residuals")
        delta_data_model = np.concatenate(
            [
                (self.bandpowers[spec] - db_model[spec])[self.spec_bin_min[i]-1:self.spec_bin_max[i]]
                for i,spec in enumerate(self.spectra_to_fit)
            ]
        )

        # Compute chisq
        self.log.debug("Compute chi2")
        chi2 = delta_data_model @ self._inv_bpcov @ delta_data_model

        self.log.debug(f"SPT3G chi2/ndof = {chi2:.14f}/{len(delta_data_model)}")
        return chi2
        

    def loglike(self, dl_cmb, **params):
        chi2 = self.compute_chi2( dl_cmb, **params)
        return -0.5 * chi2

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True, units="muK2")
        return self.loglike(
            {"TT": Cls.get("tt"), "TE": Cls.get("te"), "EE": Cls.get("ee")}, **data_params
        )

    def dof( self):
        return len(self._inv_bpcov)


    # Super sample lensing
    # Based on Manzotti et al. 2014 (https://arxiv.org/pdf/1401.7992.pdf) Eq. 32
    # Applies correction to the spectrum and returns the correction slotted into the fg array
    def SuperSampleLensing(self, kappa, Dl_theory):

        # Grab ells
        ells = np.arange(self.lmin, self.lmax + 1)

        # Grab Cl derivative
        Cl_derivative = np.gradient( Dl_theory * 2 * np.pi / (ells * (ells + 1)))

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
    def AberrationCorrection(self, ab_coeff, Dl_theory):
        # AC = beta*l(l+1)dCl/dln(l)/(2pi)

        # Grab ells
        ells = np.arange(self.lmin, self.lmax + 1)

        # Grab Cl derivative
        Cl_derivative = np.gradient( Dl_theory * 2 * np.pi / (ells * (ells + 1)))

        # Calculate aberration correction
        # (In Cl space) AC = -coeff*dCl/dln(l) = -coeff*l*dCl/dl
        # where coeff contains the boost amplitude and direction (beta*<cos(theta)> in Jeong+ 13)
        aberration_correction = -ab_coeff * Cl_derivative * ells
        aberration_correction = (
            aberration_correction * ells * (ells + 1) / (2 * np.pi)
        )  # Convert to Dl

        return aberration_correction


    # Calibration
    # Theory is scaled by the inverse
    def InternalCalibration(self, mode, calT, calE):

        if mode == 'TT':
            cal = (calT[0]        ) * (calT[1]        )
        if mode =='TE':
            cal = (calT[0]        ) * (calT[1]*calE[1])
        if mode =='EE':
            cal = (calT[0]*calE[0]) * (calT[1]*calE[1])

        return cal

    # Leakage
    # Get Temperature-to-Polarisation leakage according to Eq.14 in Camphuis et al. 2025
    def T2PLeakage(self, sigmas, epsilons, Dl_sky):
        ells = np.arange(self.lmin, self.lmax+1)

        leak = {'TT':[],'TE':[],'EE':[]}
        sig1,sig2 = sigmas
        eps1,eps2 = epsilons

        leak['TT'] = 0.*Dl_sky['TT']
        leak['TE'] = eps2 * sig2**2 * ells**2 * Dl_sky['TT']
        leak['EE'] = eps1 * sig1**2 * ells**2 * Dl_sky['TE'] + \
                     eps2 * sig2**2 * ells**2 * Dl_sky['ET'] + \
                     eps1 * eps2 * sig1**2 * sig2**2 * ells**4 * Dl_sky['TT']

        return leak

    # Beam eigenmodes
    # Beam error modes correction to propagate the error on the effective beam measurement
    def BeamEigenModes(self, freqs, betas):
        ells = np.arange(self.lmin, self.lmax+1)
        Nmodes = len(betas)

        BeamModesCorrection = (
            (1 + betas @ self.beam_eigenmodes[freqs[0]][:Nmodes,ells]) *
            (1 + betas @ self.beam_eigenmodes[freqs[1]][:Nmodes,ells])
            )

        return BeamModesCorrection

    # Beam error modes correction to propagate the error on the effective beam measurement
    def PolarizedBeam(self, mode, freqs, betas):
        ells = np.arange(self.lmin, self.lmax+1)

        #compute polar beam
        beam_pol = {}
        for freq,beta in zip(freqs,betas):
            beam_pol[freq] = self.beam_main_temperature[freq] + beta*(1. - self.beam_main_temperature[freq])

        #normalized at l=800
        for freq in freqs:
            beam_pol[freq] /= beam_pol[freq][800]

        if mode == 'TT':
            PolBeamCorrection = 1.
        elif mode == 'TE':
            PolBeamCorrection = beam_pol[freqs[1]][ells]
        elif mode == 'EE':
            PolBeamCorrection = beam_pol[freqs[0]][ells] * \
                                beam_pol[freqs[1]][ells]
        else:
            raise LoggedError( self.log, f"Wrong cross-spectrum for Polarized beam ({mode})")

        return PolBeamCorrection

    # Beam eigenmodes
    # Beam error modes correction to propagate the error on the effective beam measurement
    def ModesAndPolarizedBeam(self, mode, freqs, beta_pol, beta_modes):
        ells = np.arange(self.lmin, self.lmax+1)

        #compute modes correction
        Nmodes = len(beta_modes)
        BeamModesCorrection = {fq: 1 + beta_modes @ self.beam_eigenmodes[fq][:Nmodes] for fq in freqs}

        #compute polar beam
        BeamPolCorrection = {fq:
                             self.beam_main_temperature[fq] + beta*(BeamModesCorrection[fq] - self.beam_main_temperature[fq]) /
                             (self.beam_main_temperature[fq][800] + beta*(1 - self.beam_main_temperature[fq][800]))
                             for fq,beta in zip(freqs,beta_pol)}

        if mode == 'TT':
            BeamCorrection = BeamModesCorrection[freqs[0]][ells] * BeamModesCorrection[freqs[1]][ells]
        elif mode == 'TE':
            BeamCorrection = BeamModesCorrection[freqs[0]][ells] * BeamPolCorrection[freqs[1]][ells]
        elif mode == 'EE':
            BeamCorrection = BeamPolCorrection[freqs[0]][ells] * BeamPolCorrection[freqs[1]][ells]
        else:
            raise LoggedError( self.log, f"Wrong cross-spectrum for Polarized beam ({mode})")

        return BeamCorrection


class TTTEEE(SPT3G_D1_Lik):
    r"""
    Likelihood for Camphuis et al. 2025
    SPT-3G D1 95, 150, 220GHz bandpowers, l=300-4000, TT/TE/EE
    """


class TT(SPT3G_D1_Lik):
    r"""
    Likelihood for Camphuis et al. 2025
    SPT-3G D1 95, 150, 220GHz bandpowers, l=300-3000, TT
    """


class TE(SPT3G_D1_Lik):
    r"""
    Likelihood for Camphuis et al. 2025
    SPT-3G D1 95, 150, 220GHz bandpowers, l=300-4000, TE
    """


class EE(SPT3G_D1_Lik):
    r"""
    Likelihood for Camphuis et al. 2025
    SPT-3G D1 95, 150, 220GHz bandpowers, l=300-4000, EE
    """
