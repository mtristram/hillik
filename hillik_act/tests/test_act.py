import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "ACT_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.01040,
    "As": 2.16185225e-09,
    "ombh2": 0.02241,
    "omch2": 0.1188,
    "ns": 0.9686,
    "Alens": 1.0,
    "tau": 0.060,
}

calib_params = {
    "cal_ACT": 1.0,
    "poleff_ACT_98": 1.0,
    "poleff_ACT_150": 1.0,
    "cal_ACT_98": 1.0,
    "cal_ACT_150": 1.0,
    "leak_ACT_98": 1.0,
    "leak_ACT_150": 1.0,
}

extgal_params = {
    "Acib": 1.95,
    "Atsz": 2.16,
    "Aksz": 5.1,
    "xi": 0.5,
    "beta_cib": 1.5,
}

fg_ps_params = {
    "wide.TT": dict(
        Aps_ACTw_98x98=121.2,
        Aps_ACTw_98x150=53.8,
        Aps_ACTw_150x150=31.2,
        ),
    "deep.TT": dict(
        Aps_ACTd_98x98=20.2,
        Aps_ACTd_98x150=11.1,
        Aps_ACTd_150x150=11.8,
        ),
}

fg_dust_params = {
    "wide.TT": dict(
        Adust_ACTw_98T  = 1.24,
        Adust_ACTw_150T = 1.95,
        ),
    "wide.EE": dict(
        Adust_ACTw_98P  = 0.01,
        Adust_ACTw_150P = 0.02,
        ),
    "deep.TT": dict(
        Adust_ACTd_98T  = 2.9,
        Adust_ACTd_150T = 1.0,
        ),
    "deep.EE": dict(
        Adust_ACTd_98P  = 0.01,
        Adust_ACTd_150P = 0.02,
        ),
}

nuisance_params = {
    "wide.TT": {**fg_dust_params['wide.TT'],**extgal_params,**fg_ps_params['wide.TT']},
    "wide.EE": {**fg_dust_params['wide.EE']},
    "wide.TE": {**fg_dust_params['wide.TT'],**fg_dust_params['wide.EE']},
    "wide.TTTEEE": {**fg_dust_params['wide.TT'],**extgal_params,**fg_ps_params['wide.TT'],**fg_dust_params['wide.EE']},
    "deep.TT": {**fg_dust_params['deep.TT'],**extgal_params,**fg_ps_params['deep.TT']},
    "deep.EE": {**fg_dust_params['deep.EE']},
    "deep.TE": {**fg_dust_params['deep.TT'],**fg_dust_params['deep.EE']},
    "deep.TTTEEE": {**fg_dust_params['deep.TT'],**extgal_params,**fg_ps_params['deep.TT'],**fg_dust_params['deep.EE']},
    }


#chi2s = {"wide.TT": 60.75,"deep.TT": 48.058}   #ell>2000
#chi2s = {
#    "wide.TT":  60.75, "deep.TT": 48.058,
#    "wide.EE": 177.98, "deep.EE": 126.33,
#    "wide.TE": 126.34, "deep.TE": 111.82,
#    "wide.TTTEEE": 364.04, "deep.TTTEEE": 293.56
#    }
chi2s = {
    "wide.TT": 149.00, "deep.TT": 145.89,
    "wide.EE": 229.25, "deep.EE": 157.26,
    "wide.TE": 221.21, "deep.TE": 203.97,
    "wide.TTTEEE": 585.96, "deep.TTTEEE": 542.08
    }


class ACTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install({"likelihood": {f"hillik_act.{mode}": None}}, path=packages_path)

##     def test_act(self):
##         import camb
##         import hillik_act.wide, hillik_act.deep
        
##         #camb
##         camb_cosmo = cosmo_params.copy()
##         camb_cosmo.update({"lmax": 6000, "lens_potential_accuracy": 1})
##         pars = camb.set_params(**camb_cosmo)
##         results = camb.get_results(pars)
##         powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##         dl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        
##         for mode, chi2 in chi2s.items():
##             if "wide" in mode: _act = getattr(hillik_act.wide, mode.split('.')[1])({"packages_path": packages_path})
##             if "deep" in mode: _act = getattr(hillik_act.deep, mode.split('.')[1])({"packages_path": packages_path})
##             loglike = _act.loglike(dl_dict, **{**calib_params,**nuisance_params[mode]})
##             self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        riri = {}
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {f"hillik_act.{mode}": None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params,**nuisance_params[mode],**calib_params},
                "packages_path": packages_path,
            }

            model = get_model(info)
#            riri[mode] = -2 * model.loglikes({})[0][0])
#        print(riri)
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
