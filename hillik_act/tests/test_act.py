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
    "ACT_cal": 1.0,
    "ACT_pe_98": 1.0,
    "ACT_pe_150": 1.0,
    "ACT_cal_98": 1.0,
    "ACT_cal_150": 1.0,
    "ACT_leak_98": 1.0,
    "ACT_leak_150": 1.0,
}

extgal_params = {
    "Acib": 3.6,
    "Atsz": 5.8,
    "Aksz": 0.01,
    "xi": 0.2,
    "beta_cib": 1.5,
}

fg_ps_params = {
    "wide.TT": dict(
        ACTw_radio_ps=22.5,
        ACTw_cib_ps=6.8,
        ),
    "deep.TT": dict(
        ACTd_radio_ps=3.7,
        ACTd_cib_ps=5.8,
        ),
}

fg_dust_params = {
    "wide.TT": dict(
        ACTw_AdustT = 120.,
        ),
    "wide.EE": dict(
        ACTw_AdustP = 10.,
        ),
    "deep.TT": dict(
        ACTd_AdustT = 60.,
        ),
    "deep.EE": dict(
        ACTd_AdustP = 5.,
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


chi2s = {
    "wide.TT": 177.8, "deep.TT": 154.12,
    "wide.EE": 232.3, "deep.EE": 158.06,
    "wide.TE": 226.4, "deep.TE": 205.38,
    "wide.TTTEEE": 617.9, "deep.TTTEEE": 549.34
    }

class ACTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install({"likelihood": {f"hillik_act.{mode}": None}}, path=packages_path)

#    def test_camb(self):
#        import camb
        import hillik_act.wide, hillik_act.deep
#        
#        camb_cosmo = cosmo_params.copy()
#        camb_cosmo.update({"lmax": 6000, "lens_potential_accuracy": 1})
#        pars = camb.set_params(**camb_cosmo)
#        results = camb.get_results(pars)
#        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
#        dl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
#        
#        for mode, chi2 in chi2s.items():
#            if "wide" in mode: _act = getattr(hillik_act.wide, mode.split('.')[1])({"packages_path": packages_path})
#            if "deep" in mode: _act = getattr(hillik_act.deep, mode.split('.')[1])({"packages_path": packages_path})
#            loglike = _act.loglike(dl_dict, **{**calib_params,**nuisance_params[mode]})
##            print( f"CAMB/{mode}: {-2*loglike}")
#            self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {f"hillik_act.{mode}": None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params,**nuisance_params[mode],**calib_params},
                "packages_path": packages_path,
            }
            model = get_model(info)
#            print( f"COBAYA/{mode}: {-2*model.loglikes({})[0][0]}")
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
