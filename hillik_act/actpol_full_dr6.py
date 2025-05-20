""".. module:: ACT_full_DR6

:Synopsis: Definition of python-native CMB likelihood for ACT likelihood.
Adapted from Fortran likelihood code
https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.cfm
full ACT DR6 spectra at 90, 150, 220 in temperature and polarization

:Author: Matthieu Tristram

"""
import os
from typing import Optional, Sequence

import hillik_foregrounds as hfg
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

import sacc


#not used for ACT
maps = ["dr6_pa4_f220","dr6_pa5_f090","dr6_pa5_f150","dr6_pa6_f090","dr6_pa6_f150"]
feff = {
    "tsz": {m:int(m[-3:]) for m in maps},
    "dust": {m:int(m[-3:]) for m in maps},
    "radio": {m:int(m[-3:]) for m in maps},
    "cib": {m:int(m[-3:]) for m in maps},
    "sync": {m:int(m[-3:]) for m in maps},
    }



class ACTDR6Likelihood(InstallableLikelihood):
    install_options = {
        "download_url": "https://portal.nersc.gov/project/act/dr6_data/dr6_data.tar.gz",
        "data_path": "ACTDR6MFLike",
    }
    type = "CMB"

    fgds_folder: Optional[str] = "foregrounds"
    data_folder: Optional[str] = "ACTDR6MFLike/v1.0"
    input_file: Optional[str]  = "dr6_data.fits"

    lmin: Optional[int] = 2
    lmax: Optional[int] = 8500
    BoltzmannLmax: Optional[str] = 9000

    #----------------------------------------------------------------
    # likelihood terms from ACT data
    #----------------------------------------------------------------

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
        self.survey = "ACT"
        self.log.debug( f"Survey = {self.survey}")

        #all info in the dict spectra: experiments, polarization, scales, ell, dl, bpw
        self.spectra = self.data["spectra"]
        self.maps = self.data["experiments"]
        default_cuts = self.defaults

        #check modes
        if "TE" in self.foregrounds:
            self.foregrounds['ET'] = self.foregrounds['TE']
        self._is_mode = {mode: mode in self.defaults["polarizations"] for mode in ["TT", "TE", "ET", "EE"]}
        self.log.debug("mode = {}".format(self._is_mode))

        #-----------------------------------------------
        #load spectrum
        #-----------------------------------------------
        data = sacc.Sacc.load_fits( os.path.join(self.data_folder, self.input_file))

        def get_cl_name(pol,exp1,exp2):
            pol_dict = {"T": "0", "E": "e", "B": "b"}
            ext_dict = {"T": "0", "E": "2", "B": "2"}
            p1, p2 = pol
            tname1 = exp1+"_s"+ext_dict[p1]
            tname2 = exp2+"_s"+ext_dict[p2]

            if p2 == "T":
                dt = "cl_" + pol_dict[p2] + pol_dict[p1]
            else:
                dt = "cl_" + pol_dict[p1] + pol_dict[p2]
            return dt, tname1, tname2

        # read bandpowers
        select_ind = []
        for spec in self.spectra:
            spec["polarizations"] = spec.get("polarizations", default_cuts["polarizations"]).copy()
            for pol in spec["polarizations"]:
                spec[pol] = {}
                m1,m2 = spec["experiments"]

                #redefine lmin/lmax if global set
                if spec["scales"][pol][0] < self.lmin: spec["scales"][pol][0] = self.lmin
                if spec["scales"][pol][1] > self.lmax: spec["scales"][pol][1] = self.lmax
                lmin,lmax = spec["scales"][pol]

                dt,exp1,exp2 = get_cl_name(pol,m1,m2)
                ls,dls = data.get_ell_cl(dt,exp1,exp2)
                spec[pol]['leff'] = np.array([l for l,dl in zip(ls,dls) if l>=lmin and l<=lmax])
                spec[pol]['dl']   = np.array([dl for l,dl in zip(ls,dls) if l>=lmin and l<=lmax])
                ind = data.indices( dt, (exp1,exp2), ell__gt=lmin, ell__lt=lmax)
                spec[pol]["bpw"] = data.get_bandpower_windows(ind)
                select_ind += list(ind)
                self.log.debug( f"{spec['experiments']} {pol}: {len(ind)}bins [{lmin},{lmax}]")

        # Read Covariance Matrix (6840,6840)
        covmat = data.covariance.covmat[select_ind,:][:,select_ind]
        self.log.debug( f"matrix size = [{len(select_ind)}]")

        #invert covmat
        self.inv_cov = np.linalg.inv( covmat)
        self.logp_const = np.log(2 * np.pi) * (-len(select_ind) / 2) - 0.5*np.linalg.slogdet(covmat)[1]

        #lmin,lmax from the bandpass
        self.lmin_bpw = self.BoltzmannLmax
        self.lmax_bpw = 2
        for spec in self.spectra:
            for pol in spec["polarizations"]:
                lmin,lmax = min(spec[pol]["bpw"].values),max(spec[pol]["bpw"].values)
                self.lmin_bpw = min(self.lmin_bpw,lmin)
                self.lmax_bpw = max(self.lmax_bpw,lmax)
        self.log.debug( f"lrange = [{self.lmin_bpw},{self.lmax_bpw}]")

        #get frequencies
        self.frequencies = [int(m[-3:]) for m in self.maps]

        # Read beams (same info in T or E)
        self.beams = {}
        for m in self.maps:
            bpl = data.tracers[m+'_s0']
            self.beams[m] = dict( nu=bpl.nu,
                                  bandpass=bpl.bandpass,
                                  beam=bpl.beam[:self.lmax_bpw+1,:]/bpl.beam[0,:][np.newaxis,...]) #normalize to 1 at l=0
#            self.beams['T'][m] = data.tracers[m+'_s0']
#            self.beams['E'][m] = data.tracers[m+'_s2']

        #-----------------------------------------------
        # Init foreground model
        #-----------------------------------------------
        self.fgs = {'TT':[],'TE':[],'ET':[],'EE':[]}
        for pol,fgs in self.fgs.items():
            xfreq = [tuple(spec['experiments']) for spec in self.spectra]

            if pol == 'ET': pol = 'TE'
            if self._is_mode[pol]:
                for name in self.foregrounds[pol].keys():
                    if not hasattr(hfg,name):
                        raise LoggedError(self.log, "Unkown foreground model '%s'!", name)

                    self.log.debug("Adding '{}' foreground for {}".format(name,pol))
                    kwargs = dict(lmax=self.lmax_bpw, cross=xfreq, mode=pol, survey=self.survey, beams=self.beams)
#                    kwargs = dict(lmax=self.lmax_bpw, cross=xfreq, mode=pol, survey=self.survey, feff=feff)
                    if name == "dust": kwargs['lnorm'] = 500
                    if isinstance(self.foregrounds[pol][name], str):
                        kwargs["filename"] = os.path.join(self.fgds_folder, self.foregrounds[pol][name])
                    elif name == "szxcib":
                        filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["tsz"])
                        filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.fgds_folder, self.foregrounds["TT"]["cib"])
                        kwargs["filenames"] = (filename_tsz,filename_cib)
                    fgs.append(getattr(hfg,name)(**kwargs))
        
        self.log.info("Initialized!")


    def compute_chi2(self, dl_cmb, **params):
        """
        dl_cmb: Dl [muK2]
        """

        #get foregrounds
        dl_fg = {pol:self._compute_all_fg( fgs, params) for pol,fgs in self.fgs.items()}

        # Get residual
        delta_dl = []
        for ispec,spec in enumerate(self.spectra):
            exp1,exp2 = spec["experiments"]
            for pol in spec["polarizations"]:
                bpw = spec[pol]["bpw"]
                
                # Sum CMB and FG
                X_model = bpw.weight.T@(dl_cmb[pol][bpw.values]+dl_fg[pol][ispec,bpw.values])
                
                # apply calibration
                X_model  /= self._calibration(params,pol,exp1,exp2)
                
                # compute residual
                delta_dl += list(spec[pol]["dl"] - X_model)

        #chi2
        chi2 = self._fast_chi_squared(self.inv_cov, delta_dl)

        self.log.debug(f"Χ² = {chi2} / {len(delta_dl)}")

        return chi2

    def get_requirements(self):
        requirements = dict(Cl={mode:self.BoltzmannLmax for mode in ["tt","te","et","ee"]})
        return requirements

    def logp(self, **params_values):
        dl = self.provider.get_Cl(units="muK2", ell_factor=True)
        return self.loglike(dl, **params_values)

    def loglike(self, dl_cmb, **params):
        logp = -0.5 * self.compute_chi2( {s.upper():dl for s,dl in dl_cmb.items()}, **params) + self.logp_const
        self.log.debug(f"Log-likelihood value computed = {logp})")
        return logp


    # Calibration
    # Data is scaled as: TT: c1*c2, TE: c1*c2*p2, EE: c1*p1*c2*p2
    # Theory is scaled by the inverse
    def _calibration(self, params, pol, exp1, exp2):
        surv = self.survey
        cal = params[f'{surv}_cal']
        p1,p2 = pol        
        ct1 = params[f'{surv}_cal_{exp1}']
        ct2 = params[f'{surv}_cal_{exp2}']
        if p1 == 'E': ct1 = ct1*params[f'{surv}_pe_{exp1}']
        if p2 == 'E': ct2 = ct2*params[f'{surv}_pe_{exp2}']
        return cal*cal*ct1*ct2


    def _compute_all_fg( self, fg_list, params):
        dl_fg = np.zeros( (len(self.spectra), self.lmax_bpw+1) )

        for fg in fg_list:
            dl_fg += fg.compute_dl( params)

        return dl_fg


class TT(ACTDR6Likelihood):
    """
    CMB likelihood with ACTpol DR6 full dataset
    """

class TE(ACTDR6Likelihood):
    """
    CMB likelihood with ACTpol DR6 full dataset
    """

class EE(ACTDR6Likelihood):
    """
    CMB likelihood with ACTpol DR6 full dataset
    """

class TTTEEE(ACTDR6Likelihood):
    """
    CMB likelihood with ACTpol DR6 full dataset
    """

