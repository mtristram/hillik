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
    "TThighl": {
        "SPT_cal": 1.0,
        "SPT_cal_95": 1.0,
        "SPT_cal_150": 1.0,
        "SPT_cal_220": 1.0,
        "FTS_calibration_error": 1.0,
        },
    "TT": {
        "SPT3G_cal": 1.0,
        "SPT3G_cal_T90": 1.0,
        "SPT3G_cal_T150": 1.0,
        "SPT3G_cal_T220": 1.0,
        "SPT3G_cal_E90": 1.0,
        "SPT3G_cal_E150": 1.0,
        "SPT3G_cal_E220": 1.0,
        "kappa": 0.0
        },
    "EE": {
        "SPT3G_cal": 1.0,
        "SPT3G_cal_T90": 1.0,
        "SPT3G_cal_T150": 1.0,
        "SPT3G_cal_T220": 1.0,
        "SPT3G_cal": 1.0,
        "SPT3G_cal_E90": 1.0,
        "SPT3G_cal_E150": 1.0,
        "SPT3G_cal_E220": 1.0,
        "kappa": 0.0
        },
    "TE": {
        "SPT3G_cal": 1.0,
        "SPT3G_cal_T90": 1.0,
        "SPT3G_cal_T150": 1.0,
        "SPT3G_cal_T220": 1.0,
        "SPT3G_cal_E90": 1.0,
        "SPT3G_cal_E150": 1.0,
        "SPT3G_cal_E220": 1.0,
        "kappa": 0.0
        },
    "TTTEEE": {
        "SPT3G_cal": 1.0,
        "SPT3G_cal_T90": 1.0,
        "SPT3G_cal_T150": 1.0,
        "SPT3G_cal_T220": 1.0,
        "SPT3G_cal_E90": 1.0,
        "SPT3G_cal_E150": 1.0,
        "SPT3G_cal_E220": 1.0,
        "kappa": 0.0
        },
}

fg_params = {
    "TThighl": dict(
        Acib=1.5,
        Atsz=4.5,
        Aksz=1.5,
        xi=0.1,
        beta_cib=1.5,
        beta_radio=-0.8,
        SPT_radio_TT=1.,
        SPT_cib_ps=8.6,
        SPT_AdustTT=5.46,
        ),
    "TT": dict(
        Acib=1.5,
        Atsz=4.5,
        Aksz=1.5,
        xi=0.1,
        beta_cib=1.5,
        beta_radio=-0.8,
        SPT3G_radio_TT=10.,
        SPT3G_cib_ps=7.,
        SPT3G_AdustTT=1.15,
        ),
    "TE": dict(
        SPT3G_AdustTE=0.073,
        SPT3G_radio_TE=1.,
        ),
    "EE": dict(
        SPT3G_AdustEE=0.031,
        SPT3G_radio_EE=1.,
        ),
    "TTTEEE": dict(
        SPT3G_AdustTT=1.15,
        SPT3G_AdustTE=0.073,
        SPT3G_AdustEE=0.031,
        Acib=1.5,
        Atsz=4.5,
        Aksz=1.5,
        xi=0.1,
        beta_cib=1.5,
        beta_radio=-0.8,
        SPT3G_cib_ps=7.,
        SPT3G_radio_TT=10.,
        SPT3G_radio_TE=1.,
        SPT3G_radio_EE=1.,
        )
    }

chi2s = {"TThighl":889.096, "TT":1482.95, "EE":431.69, "TE":683.33, "TTTEEE":2607.78}


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
                "params": {**cosmo_params, **calib_params[mode], **fg_params[mode]},
                "packages_path": packages_path,
            }
            
            model = get_model(info)
#            print( f"COBAYA/{mode}: {-2*model.loglikes({})[0][0]}")
            self.assertLess( abs(-2*model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
