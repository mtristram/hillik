"""
.. module:: HiLLiPoP for Hillik

:Synopsis: Likelihood class for Planck PR4 to be used alongside ACT and SPT
:Author: Matthieu Tristram

Hillipop is a multifrequency CMB likelihood for Planck data.

The likelihood is a spectrum-based Gaussian approximation for
cross-correlation spectra from Planck 100, 143 and 217GHz
split-frequency maps, with semi-analytic estimates of the Cl
covariance matrix based on the data. The cross-spectra are debiased
from the effects of the mask and the beam leakage using Xpol before
being compared to the model, which includes CMB and foreground
residuals. They cover the multipoles from l=30 to l=2500.

This Hillik version makes the foreground treatment match between Planck, ACT,
and SPT, and ensures there is not multipole overlap between Planck and ACT.

"""

import glob
import logging
import os
import re
from itertools import combinations
from typing import Optional
from copy import deepcopy

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import data_path, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.mpi import is_main_process

import hillik_foregrounds as fg
from . import bins


#predefined binning
lower_l = np.arange( 30,  251,  1)  # unbinned
upper_l = np.arange(251, 2500, 10)  # binned
bin_lmins = np.concatenate((lower_l, upper_l))
bin_lmaxs = np.concatenate((lower_l, upper_l + 9))

#effective frequencies
maps = ["100A", "100B", "143A", "143B", "217A", "217B"]
feff = {
    "tsz":   {100:100.2, 143:143.0, 217:222.0},
    "dust":  {100:105.2, 143:148.5, 217:228.1}, #alpha=4 from [Planck 2013 IX]
    "cib":   {100:105.2, 143:148.5, 217:228.1}, #alpha=4 from [Planck 2013 IX]
    "radio": {100:100.4, 143:140.5, 217:218.6},
    "sync":  {100:100.4, 143:140.5, 217:218.6},
    }


# ------------------------------------------------------------------------------------------------
# Likelihood
# ------------------------------------------------------------------------------------------------

data_url = "https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods"


class _HillipopLikelihood(InstallableLikelihood):
    type = "CMB"

    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "planck_2020/hillipop"
    multipoles_range_file: Optional[str]
    xspectra_basename: Optional[str]
    covariance_matrix_file: Optional[str]
    foregrounds: Optional[dict]
    lrange: Optional[dict] = None

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to Hillipop data. Set the likelihood property "
                f"'path' or the common property '{packages_path_input}'.",
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, data_path)
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                f"The 'data_folder' directory does not exist. "
                f"Check the given path [{self.data_folder}].",
            )
        self.fgds_folder = os.path.join(data_file_path, self.fgds_folder)
        if not os.path.exists(self.fgds_folder):
            raise LoggedError(
                self.log,
                f"The 'fgds_folder' directory does not exist. "
                f"Check the given path [{self.fgds_folder}].",
            )

        self.frequencies = [100, 100, 143, 143, 217, 217]
        self._mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
        self._nmap = len(self.frequencies)
        self._nfreq = len(np.unique(self.frequencies))
        self._nxfreq = self._nfreq * (self._nfreq + 1) // 2
        self._nxspec = self._nmap * (self._nmap - 1) // 2
        self._xspec2xfreq = self._xspec2xfreq()
        self._xfreq_labels = self._xfreq_labels()
        self.log.debug(f"frequencies = {self.frequencies}")

        # Define the hillik-survey
        self.survey = "PLK"

        # Get likelihood name and add the associated mode
        likelihood_name = self.__class__.__name__
        likelihood_modes = [likelihood_name[i:i+2] for i in range(0, len(likelihood_name), 2)]
        self._is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE", "BB"]}
        self._is_mode["ET"] = self._is_mode["TE"]
        self.log.debug(f"mode = {self._is_mode}")

        # Multipole ranges and binning
        filename = os.path.join(self.data_folder, self.multipoles_range_file)
        self._lmins, self._lmaxs = self._set_multipole_ranges(filename)
        self.lmin = min(min(self._lmins[mode]) for mode, is_mode in self._is_mode.items() if is_mode)
        self.lmax = max(max(self._lmaxs[mode]) for mode, is_mode in self._is_mode.items() if is_mode)
        self.wf = bins.Bins(bin_lmins, bin_lmaxs)

        # Inverted Covariance matrix
        filename = os.path.join(self.data_folder, self.covariance_matrix_file)
        self._invkll = self._read_invcovmatrix(filename)
        if self.lrange:
            self._cut_lrange()
        self._invkll = self._invkll.astype('float32')

        # Precompute binning operators for _select_spectra
        self._bin_p = {}
        for mode in ["TT", "TE", "EE"]:
            if not self._is_mode[mode]:
                continue
            for xf in range(self._nxfreq):
                lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
                lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
                wf = deepcopy(self.wf)
                wf.cut_binning(lmin, lmax)
                p, _ = wf._bin_operators(Dl=False)
                self._bin_p[(mode, xf)] = p

        # Data
        basename = os.path.join(self.data_folder, self.xspectra_basename)
        self._dldata = self._read_dl_xspectra(basename)

        # Weights
        dlsig = self._read_dl_xspectra(basename, hdu=2)
        for m,w8 in dlsig.items(): w8[w8==0] = np.inf
        self._dlweight = {k:1/v**2 for k,v in dlsig.items()}

        # Foregrounds
        self.fgs = {tag:[] for tag, is_tag in self._is_mode.items() if is_tag}  # list of foregrounds per mode [TT,EE,TE,ET]
        if 'TE' in self.foregrounds: self.foregrounds['ET'] = self.foregrounds['TE']
        for tag,fgs in self.fgs.items():
            for name in self.foregrounds[tag].keys():
                if not hasattr( fg, name):
                    raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                self.log.debug("Adding '{}' foreground for {}".format(name,tag))
                kwargs = dict(lmax=self.lmax, cross=list(combinations(self.frequencies, 2)), mode=tag, survey=self.survey, feff=feff)
                if isinstance(self.foregrounds[tag][name], str):
                    kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds[tag][name])
                elif name == "szxcib":
                    kwargs["filename_tsz"] = self.foregrounds[tag]["tsz"] and os.path.join(self.fgds_folder, self.foregrounds[tag]["tsz"])
                    kwargs["filename_cib"] = self.foregrounds[tag]["cib"] and os.path.join(self.fgds_folder, self.foregrounds[tag]["cib"])
                fgs.append(getattr(fg,name)(**kwargs))

        if is_main_process():
            for xs, (m1, m2) in enumerate(combinations(self._mapnames, 2)):
                logstr = f"{m1}x{m2}: "
                for mode in ['TT','EE','TE']:
                    if self._is_mode[mode]:
                        xflmin = self._lmins[mode][xs]
                        xflmax = self._lmaxs[mode][xs]
                        wf = deepcopy( self.wf)
                        wf.cut_binning( xflmin, xflmax)
                        logstr += f"{mode}[{wf.lmin:4d}-{wf.lmax:4d}]  "
                self.log.debug(logstr)

        self.log.info("Initialized!")

    def _xspec2xfreq(self):
        list_fqs = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                list_fqs.append((f1, f2))

        freqs = list(np.unique(self.frequencies))
        spec2freq = []
        for m1 in range(self._nmap):
            for m2 in range(m1 + 1, self._nmap):
                f1 = freqs.index(self.frequencies[m1])
                f2 = freqs.index(self.frequencies[m2])
                spec2freq.append(list_fqs.index((f1, f2)))

        return spec2freq

    def _xfreq_labels(self):
        freqs = list(np.unique(self.frequencies))
        labels = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                labels.append(f"{freqs[f1]}x{freqs[f2]}")
        return labels

    def _set_multipole_ranges(self, filename):
        """
        Return the (lmin,lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
        array(nmode,nxspec)
        """
        self.log.debug("Define multipole ranges")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        tags = ["TT", "EE", "BB", "TE"]
        lmins = {}
        lmaxs = {}
        with fits.open( filename) as hdus:
            for hdu in hdus[1:]:
                tag = hdu.header['spec']
                lmins[tag] = hdu.data.LMIN
                lmaxs[tag] = hdu.data.LMAX
                if self._is_mode[tag]:
                    self.log.debug(f"{tag}")
                    self.log.debug(f"lmin: {lmins[tag]}")
                    self.log.debug(f"lmax: {lmaxs[tag]}")
        lmins["ET"] = lmins["TE"]
        lmaxs["ET"] = lmaxs["TE"]

        return lmins, lmaxs

    def _read_dl_xspectra(self, basename, hdu=1):
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl (TT,EE,TE,ET) in muK^2
        """
        self.log.debug("Reading cross-spectra %s" % ("weights" if hdu == 2 else ""))

        with fits.open(f"{basename}_{self._mapnames[0]}x{self._mapnames[1]}.fits") as hdus:
            nhdu = len( hdus)
        if nhdu < hdu:
            #no sig in file, uniform weight
            self.log.info( "Warning: uniform weighting for combining spectra !")
            dldata = np.ones( (self._nxspec, 4, self.lmax+1))
        else:
            if nhdu == 1: hdu=0 #compatibility
            dldata = []
            for m1, m2 in combinations(self._mapnames, 2):
                data = fits.getdata( f"{basename}_{m1}x{m2}.fits", hdu)*1e12
                tmpcl = list(data[[0,1,3],:self.lmax+1])
                data = fits.getdata( f"{basename}_{m2}x{m1}.fits", hdu)*1e12
                tmpcl.append( data[3,:self.lmax+1])
                dldata.append( tmpcl)

        dldata = np.transpose(np.array(dldata), (1, 0, 2))
        return dict(zip(['TT','EE','TE','ET'],dldata))

    def _read_invcovmatrix(self, filename):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        self.log.debug(f"Covariance matrix file: {filename}")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        data = fits.getdata(filename)
        nel = int(np.sqrt(data.size))
        data = data.reshape((nel, nel)) / 1e24  # muK^-4
        data = 0.5 * (data + data.T)  # ensure exact symmetry

        nell = self._get_matrix_size()
        if nel != nell:
            raise ValueError(f"Incoherent covariance matrix (read:{nel}, expected:{nell})")

        return data

    def _cut_lrange(self):
        """
        Apply multipole cuts from `lrange` on the covariance matrix and lmins/lmaxs.
        """

        # build index list for covariance cutting
        self.log.debug(f"\tSet up lmin/lmax cuts for lrange:\n{self.lrange}")
        idx_offset = 0
        kept_idxs = []
        for XY in ["TT", "EE", "TE"]:
            if not self._is_mode[XY]:
                continue
            effective_lmins = self._lmins[XY].copy()
            effective_lmaxs = self._lmaxs[XY].copy()
            l_cuts = self.lrange.get(XY)
            if l_cuts is None:
                lmin_cut = self.lmax
                lmax_cut = self.lmin
            else:
                lmin_cut, lmax_cut = l_cuts

            # validate lrange
            if lmin_cut < min(self._lmins[XY]):
                raise LoggedError(self.log, f"{XY} lmin should be >= {min(self._lmins[XY])} (lmin={lmin_cut} requested)")
            if lmax_cut > max(self._lmaxs[XY]):
                raise LoggedError(self.log, f"{XY} lmax should be <= {max(self._lmaxs[XY])} (lmax={lmax_cut} requested)")

            for xf in range(self._nxfreq):
                xs_idxs = np.where(np.array(self._xspec2xfreq) == xf)[0]
                xflmin = self._lmins[XY][xs_idxs[0]]
                xflmax = self._lmaxs[XY][xs_idxs[0]]

                wf = deepcopy(self.wf)
                wf.cut_binning(xflmin, xflmax)

                mask = (wf.lmins >= lmin_cut) & (wf.lmaxs <= lmax_cut)
                kept_idxs.extend(idx_offset + np.flatnonzero(mask))
                idx_offset += wf.nbins
                if mask.any():
                    effective_lmins[xs_idxs] = wf.lmins[mask].min()
                    effective_lmaxs[xs_idxs] = wf.lmaxs[mask].max()
                else:
                    effective_lmins[xs_idxs] = max(xflmin, lmin_cut)
                    effective_lmaxs[xs_idxs] = min(xflmax, lmax_cut)

                if mask.sum() < len(mask) and is_main_process():
                    if mask.any():
                        self.log.debug(
                            f"Cutting {XY} {self._xfreq_labels[xf]} to [{lmin_cut}, {lmax_cut}]. "
                            f"Keeping {mask.sum()}/{len(mask)} bins "
                            f"(effective range [{wf.lmins[mask].min()}, {wf.lmaxs[mask].max()}])."
                        )
                    else:
                        self.log.debug(f"Removing {XY} {self._xfreq_labels[xf]} entirely ({len(mask)} bins cut).")

            # integrate lrange into _lmins/_lmaxs
            X, Y = XY
            for YX in {XY, Y+X}:
                if l_cuts is not None:
                    self._lmins[YX] = effective_lmins.copy()
                    self._lmaxs[YX] = effective_lmaxs.copy()
                else:
                    self._is_mode[YX] = False
                    self._lmins.pop(YX, None)
                    self._lmaxs.pop(YX, None)
        used_modes = [XY for XY in ["TT", "EE", "TE"] if self._is_mode[XY]]
        self.lmin = min(min(self._lmins[XY]) for XY in used_modes)
        self.lmax = max(max(self._lmaxs[XY]) for XY in used_modes)

        # early exit if nothing was cut
        if len(kept_idxs) == self._invkll.shape[0]:
            return

        # invert, cut, re-invert covariance matrix
        self.log.debug("\tInvert invkll matrix")
        kll = np.linalg.inv(self._invkll)
        kll = 0.5 * (kll + kll.T)  # restore symmetry after numerical inversion
        kll = kll[kept_idxs,:][:,kept_idxs]
        self.log.debug("\tInvert kll matrix")
        self._invkll = np.linalg.inv(kll)
        self._invkll = 0.5 * (self._invkll + self._invkll.T)

    def _get_matrix_size(self):
        """
        Compute covariance matrix size given activated mode
        Return: number of multipole
        """
        nell = 0

        # TT,EE,TEET
        for m in ["TT", "EE", "TE"]:
            if self._is_mode[m]:
                for xf in range(self._nxfreq):
                    lmin = self._lmins[m][self._xspec2xfreq.index(xf)]
                    lmax = self._lmaxs[m][self._xspec2xfreq.index(xf)]
                    wf = deepcopy( self.wf)
                    wf.cut_binning( lmin, lmax)
                    nell += wf.nbins

        return nell

    def _select_spectra(self, cl, mode):
        """
        Cut spectra given Multipole Ranges and flatten
        Return: list
        """
        acl = np.asarray(cl)
        xl = []
        for xf in range(self._nxfreq):
            p = self._bin_p[(mode, xf)]
            minlmax = min(acl[xf].shape[0], p.shape[1])
            xl += list(np.dot(acl[xf, :minlmax], p.T[:minlmax]))
        return xl

    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency
        """
        xcl = np.zeros((self._nxfreq, self.lmax + 1))
        xw8 = np.zeros((self._nxfreq, self.lmax + 1))
        for xs in range(self._nxspec):
            xcl[self._xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self._xspec2xfreq[xs]] += weight[xs]

        xw8[xw8 == 0] = np.inf
        if normed:
            return xcl / xw8
        else:
            return xcl, xw8

    def _calibration( self, mode, map1, map2, pars):

        if mode == "TT":
            cal1 = pars[f"{self.survey}_cal_{map1}"]
            cal2 = pars[f"{self.survey}_cal_{map2}"]
        elif mode == "EE":
            cal1 = pars[f"{self.survey}_cal_{map1}"]*pars[f"{self.survey}_pe_{map1}"]
            cal2 = pars[f"{self.survey}_cal_{map2}"]*pars[f"{self.survey}_pe_{map2}"]
        elif mode == "TE":
            cal1 = pars[f"{self.survey}_cal_{map1}"]
            cal2 = pars[f"{self.survey}_cal_{map2}"]*pars[f"{self.survey}_pe_{map2}"]
        elif mode == "ET":
            cal1 = pars[f"{self.survey}_cal_{map1}"]*pars[f"{self.survey}_pe_{map1}"]
            cal2 = pars[f"{self.survey}_cal_{map2}"]

        return cal1 * cal2 * pars["A_planck"] ** 2

    def _compute_residuals(self, pars, dlth, mode):
        # Nuisances
        cal = [self._calibration( mode, m1, m2, pars) for m1, m2 in combinations(self._mapnames, 2)]

        # Data
        dldata = self._dldata[mode]

        # Model
        dlmodel = [dlth[mode]] * self._nxspec
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl(pars)

        # Compute Rl = Dl - Dlth
        Rspec = np.array([dldata[xs] - dlmodel[xs]/cal[xs] for xs in range(self._nxspec)])

        return Rspec

    def dof(self):
        return len(self._invkll)

    def reduction_matrix(self, mode):
        """
        Reduction matrix (eq. 19 Dutcher21): inv(X.T*invC*X) * X.T*invC*D
        """
        X = np.zeros( (len(self.delta_dl),self.lmax+1) )
        x0 = 0
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            wf = deepcopy(self.wf)
            wf.cut_binning(lmin, lmax)
            for il, (bmin, bmax, dl) in enumerate(zip(wf.lmins, wf.lmaxs, wf.dl)):
                X[x0+il, bmin:bmax+1] = 1/dl
            x0 += wf.nbins
        
        return X

    def compute_chi2(self, dlth, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: array or arr2d
              CMB power spectrum (Dl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # Create Data Vector
        Xl = []
        if self._is_mode["TT"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, "TT")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["TT"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'TT')

        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, "EE")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["EE"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'EE')

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(params_values, dlth, "TE")
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self._dlweight["TE"], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(params_values, dlth, "ET")
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self._dlweight["ET"], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, 'TE')

        self.delta_dl = np.asarray(Xl).astype('float32')
#        chi2 = self.delta_dl @ self._invkll @ self.delta_dl
#        chi2 = self._invkll.dot(self.delta_dl).dot(self.delta_dl)
        chi2 = self._fast_chi_squared(self._invkll, self.delta_dl)

        #protect against cast float32
        alpha = 8. - np.ceil(np.log10(chi2))
        chi2 = np.float64(np.round(chi2*10**alpha))*10**(-alpha)

        self.log.debug(f"chi2/ndof = {chi2}/{len(self.delta_dl)}")
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["tt", "ee", "te"]})

    def logp(self, **params_values):
        dl = self.provider.get_Cl(ell_factor=True)
        return self.loglike(dl, **params_values)

    def loglike(self, dl, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: dict
              CMB power spectrum (Dl in µK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """
        # cl_boltz from Boltzmann (Cl in muK^2)
        dlth = {k.upper():dl[k][:self.lmax+1] for k in dl.keys()}
        dlth['ET'] = dlth['TE']

        chi2 = self.compute_chi2(dlth, **params_values)

        return -0.5 * chi2

    @classmethod
    def get_path(cls, path):
        if path.rstrip(os.sep).endswith(data_path):
            return path
        return os.path.realpath(os.path.join(path, data_path))

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs.get("data", True):
            path = cls.get_path(kwargs["path"])
            if not (
                cls.get_install_options() and os.path.exists(path) and len(os.listdir(path)) > 0
            ):
                return False
            # Test if the covariance file is there
            ext = cls.__name__
            for suffix in ["_tristram2026cut", "_PACTcut_0overlap", "_PACTcut"]:
                if ext.endswith(suffix):
                    ext = ext[:-len(suffix)]
                    break
            if ext in {"TT", "TTTEEE"}:
                ext = f"{ext}_bin"
            test_path = os.path.join(path, f"**/invfll_PR4_v4.2_{ext}.fits")
            return len(glob.glob(test_path, recursive=True)) > 0
        return True


# ------------------------------------------------------------------------------------------------


def _get_install_options(filename):
    return {"download_url": f"{data_url}/{filename}"}


class TT(_HillipopLikelihood):
    """High-L TT Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps.
    """
    install_options = _get_install_options("planck_2020_hillipop_TT_bin_v4.2.tar.gz")

class TE(_HillipopLikelihood):
    """High-L TE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps.
    """
    install_options = _get_install_options("planck_2020_hillipop_TE_v4.2.tar.gz")

class EE(_HillipopLikelihood):
    """High-L EE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps.
    """
    install_options = _get_install_options("planck_2020_hillipop_EE_v4.2.tar.gz")

class TTTEEE(_HillipopLikelihood):
    """High-L TT+TE+EE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood
    with foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz
    split-frequency maps.
    """
    install_options = _get_install_options("planck_2020_hillipop_TTTEEE_bin_v4.2.tar.gz")


class TT_tristram2026cut(TT):
    """Planck TT likelihood cut to the Planck side of the Planck-ACT split."""

class TE_tristram2026cut(TE):
    """Planck TE likelihood cut to the Planck side of the Planck-ACT split."""

class EE_tristram2026cut(EE):
    """Planck EE likelihood cut to the Planck side of the Planck-ACT split."""

class TTTEEE_tristram2026cut(TTTEEE):
    """Planck TT+TE+EE likelihood cut to the Planck side of the Planck-ACT split."""


class TT_PACTcut(TT):
    """Planck TT likelihood cut to the Planck side of the original P-ACT split."""

class TE_PACTcut(TE):
    """Planck TE likelihood cut to the Planck side of the original P-ACT split."""

class EE_PACTcut(EE):
    """Planck EE likelihood cut to the Planck side of the original P-ACT split."""

class TTTEEE_PACTcut(TTTEEE):
    """Planck TT+TE+EE likelihood cut to the Planck side of the original P-ACT split."""


class TT_PACTcut_0overlap(TT):
    """Planck TT likelihood cut to the Planck side of the zero-overlap P-ACT split."""

class TE_PACTcut_0overlap(TE):
    """Planck TE likelihood cut to the Planck side of the zero-overlap P-ACT split."""

class EE_PACTcut_0overlap(EE):
    """Planck EE likelihood cut to the Planck side of the zero-overlap P-ACT split."""

class TTTEEE_PACTcut_0overlap(TTTEEE):
    """Planck TT+TE+EE likelihood cut to the Planck side of the zero-overlap P-ACT split."""

