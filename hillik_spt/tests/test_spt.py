import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
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
    "cal_SPT": 1.0,
    "cal_SPT_95": 1.0,
    "cal_SPT_150": 1.0,
    "cal_SPT_220": 1.0,
    "FTS_calibration_error": 1.0,
}

fg_params = dict(
    Acib=2.8,
    Atsz=0.6,
    Aksz=1.5,
    xi=0.6,
    beta_cib=2.15,
    Aps_SPT_95x95=5.5,
    Aps_SPT_95x150=4.1,
    Aps_SPT_95x220=8.2,
    Aps_SPT_150x150=7.1,
    Aps_SPT_150x220=19.1,
    Aps_SPT_220x220=53.9,
    Adust_SPT_95=4.2,
    Adust_SPT_150=8.5,
    Adust_SPT_220=33.2,
)

chi2s = {"TT": 307.19}


class SPTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"hillik_spt.TT": None}}, path=packages_path)

    def test_spt(self):
        import camb
        import hillik_spt

        #camb
        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 13500, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        dl_tt = powers["total"][:,0]

        _spt = hillik_spt.TT(dict(packages_path=packages_path))
        loglike = _spt.loglike(dl_tt, **fg_params,**calib_params)
        self.assertAlmostEqual(-2 * loglike, chi2s['TT'], 2)

    def test_cobaya(self):
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {"hillik_spt.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **fg_params},
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 2)


if __name__ == "__main__":
    unittest.main()
