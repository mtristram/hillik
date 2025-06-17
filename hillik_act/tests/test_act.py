import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "ACT_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.01040,
    "As": 2.1178e-09,
    "ombh2": 0.022591,
    "omch2": 0.1238,
    "ns": 0.9666,
    "Alens": 1.0,
    "tau": 0.056,
}

ACTmaps = ["dr6_pa4_f220","dr6_pa5_f090","dr6_pa5_f150","dr6_pa6_f090","dr6_pa6_f150"]

syste_params = {
    "ACT_cal": 1.0,
    **{f"ACT_cal_{m}": 1.0 for m in ACTmaps},
    **{f"ACT_pe_{m}": 1.0 for m in ACTmaps},
    **{f"ACT_band_shift_{m}": 0. for m in ACTmaps}
}

extgal_params = {
    "Acib": 3.69,
    "Atsz": 3.35,
    "Aksz": 1.48,
    "xi": 0.088,
    "beta_cib": 1.87,
    "beta_radio": -0.78,
    "beta_dusty": 1.87,
    "ACT_cib_ps": 7.65,
    "ACT_radio_TT": 2.86,
}

fg_params = {
    'TT':{'ACT_AdustTT': 7.97, 'ACT_beta_dustTT':1.5, 'ACT_alpha_dustTT':-2.6},
    'TE':{'ACT_AdustTE': 0.42, 'ACT_beta_dustTE':1.5, 'ACT_alpha_dustTE':-2.4},
    'EE':{'ACT_AdustEE': 0.17, 'ACT_beta_dustEE':1.5, 'ACT_alpha_dustEE':-2.4, 'ACT_radio_EE': 0.0},
}


nuisance_params = {
    "TT": {**fg_params['TT'],**extgal_params},
    "EE": {**fg_params['EE']},
    "TE": {**fg_params['TE']},
    "TTTEEE": {**fg_params['TT'],**extgal_params,**fg_params['TE'],**fg_params['EE']},
    }


chi2s = {
    "TT": 3254.41,
    "EE":  979.42,
    "TE": 1925.22,
    "TTTEEE": 6133.82,
    }

class ACTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install({"likelihood": {f"hillik_act.{mode}": None}}, path=packages_path)

##     def test_camb(self):
##         import camb
##         import hillik_act
        
##         camb_cosmo = cosmo_params.copy()
##         camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
##         pars = camb.set_params(**camb_cosmo)
##         results = camb.get_results(pars)
##         powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##         dl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        
##         for mode, chi2 in chi2s.items():
##             _act = getattr(hillik_act, mode)({"packages_path": packages_path})
##             loglike = _act.loglike(dl_dict, **{**syste_params,**nuisance_params[mode]})
##             print( f"CAMB/{mode}: {-2*loglike}")
#            self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {f"hillik_act.{mode}": None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params,**nuisance_params[mode],**syste_params},
                "packages_path": packages_path,
            }
            model = get_model(info)
            print( f"COBAYA/{mode}: {-2*model.loglikes({})[0][0]}")
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
