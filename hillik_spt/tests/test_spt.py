import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)

cosmo_pars = {
    "cosmomc_theta": 0.01040,
    "As": 2.16185225e-09,
    "ombh2": 0.02241,
    "omch2": 0.1188,
    "ns": 0.9686,
    "Alens": 1.0,
    "tau": 0.060,
}

freqs = [90,150,220]

nui_pars = { "SPT3G_cal": 1.00,
             "SPT3G_kappa": 5e-6,
             **{f"SPT3G_cal_{m}":1.00 for m in freqs},
             **{f"SPT3G_pe_{m}":1.00 for m in freqs},
             **{f'SPT3G_T2P2_{fq}':0. for fq in freqs},
             **{f'SPT3G_beta_{i+1}':-0.5 for i in range(9)},
             **{f'SPT3G_beta_pol_{fq}':0.5 for fq in freqs}}

fgs_pars = {
    'TT': {
        'xi':0.26, 'Atsz':0.94, 'Acib':3.0, 'Aksz':2.3,
        'beta_cib':1.80, 'beta_dusty': 1.80, 'beta_radio': -0.8, 'T_cib':25.,
        'SPT3G_AdustTT': 1.98, 'SPT3G_alpha_dustTT':-2.53, 'SPT3G_beta_dustTT':1.5,
        'SPT3G_radio_TT': 1., 'SPT3G_cib_ps': 8., 
#           'SPT3G_ps_90x90':  10.7, 'SPT3G_ps_90x150':  8.6, 'SPT3G_ps_90x220': 16.6,
#           'SPT3G_ps_150x150':11.9, 'SPT3G_ps_150x220':32.0, 'SPT3G_ps_220x220':95.0,
        },
    'TE': {
           'SPT3G_AdustTE': 0.10, 'SPT3G_alpha_dustTE':-2.40, 'SPT3G_beta_dustTE':1.5,
        },
    'EE': {
           'SPT3G_AdustEE': 0.05, 'SPT3G_alpha_dustEE':-2.40, 'SPT3G_beta_dustEE':1.5, 
           'SPT3G_radio_EE': 0.
           }
    }
fgs_pars['TTTEEE'] = {p:v for tag in ['TT','TE','EE']  for p,v in fgs_pars[tag].items()}

chi2s = {"TTTEEE":2300.714}


class SPTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"hillik_spt.{}".format(mode): None}},
                path=packages_path,
                skip_global=True,
            )

##     def test_camb(self):
##         import camb
##         import hillik_spt

##         camb_cosmo = cosmo_params.copy()
##         for mode, chi2 in chi2s.items():
##             _spt = getattr(hillik_spt, mode)({"debug":True,"packages_path":packages_path})

##             camb_cosmo.update({"lmax": 10000, "lens_potential_accuracy": 1})
##             pars = camb.set_params(**camb_cosmo)
##             results = camb.get_results(pars)
##             powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##             cl_boltz = {k: powers["total"][:, v] for k, v in {"tt": 0, "TT": 0, "EE": 1, "TE": 3}.items()}
            
##             loglike = _spt.loglike(cl_boltz, **fg_params[mode],**calib_params[mode])
##             print( f"CAMB/{mode}: {-2*loglike}")
#            self.assertAlmostEqual(-2*loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for mode, chi2 in chi2s.items():
            info = {
                "debug": False,
                "likelihood": {"hillik_spt.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_pars, **nui_pars, **fgs_pars[mode]},
                "packages_path": packages_path,
            }
            
            model = get_model(info)
#            print( f"COBAYA/{mode}: {-2*model.loglikes({})[0][0]}")
            self.assertLess( abs(-2*model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
