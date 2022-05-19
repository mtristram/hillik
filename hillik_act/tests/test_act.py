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
    "Acib": 2.8,
    "Atsz": 0.6,
    "Aksz": 1.5,
    "xi": 0.6,
    "beta_cib": 2.15,
}

fg_params = {
    "wide": dict(
        Aps_ACTw_98x98=82.3,
        Aps_ACTw_98x150=36.1,
        Aps_ACTw_150x150=20.6,
        Adust_ACTw_98=12.1,
        Adust_ACTw_150=11.1,
        ),
    "deep": dict(
        Aps_ACTd_98x98=13.4,
        Aps_ACTd_98x150=7.5,
        Aps_ACTd_150x150=7.8,
        Adust_ACTd_98=9.7,
        Adust_ACTd_150=5.6,
        ),
}        

chi2s = {"wide": 149.27,"deep": 152.59}


class ACTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"hillik_act.wide.TT": None}}, path=packages_path)
        install({"likelihood": {"hillik_act.deep.TT": None}}, path=packages_path)

    def test_act(self):
        import camb
        import hillik_act

        #camb
        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 6000, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        dl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

        for surv, chi2 in chi2s.items():
            _act = getattr(hillik_act, surv).TT(dict(packages_path=packages_path))
            loglike = _act.loglike(dl_dict, **fg_params[surv],**extgal_params,**calib_params)
            self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        for surv, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {f"hillik_act.{surv}.TT": None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **fg_params[surv], **extgal_params},
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
