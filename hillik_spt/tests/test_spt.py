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
        "kappa": 0.0
        },
    "EE": {
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
        SPT_radio_ps=1.,
        SPT_cib_ps=6.,
        SPT_AdustT=60.,
        ),
    "TT": dict(
        Acib=1.5,
        Atsz=4.5,
        Aksz=1.5,
        xi=0.1,
        beta_cib=1.5,
        SPT3G_radio_ps=10.,
        SPT3G_cib_ps=7.,
        SPT3G_AdustT=60.,
        ),
    "TE": dict(
        SPT3G_AdustT=60.,
        SPT3G_AdustP=6.,
        ),
    "EE": dict(
        SPT3G_AdustP=6.,
        ),
    "TTTEEE": dict(
        SPT3G_AdustT=60.,
        SPT3G_AdustP=6.,
        Acib=1.5,
        Atsz=4.5,
        Aksz=1.5,
        xi=0.1,
        beta_cib=1.5,
        SPT3G_radio_ps=10.,
        SPT3G_cib_ps=7.,
        )
    }

chi2s = {"TThighl":601.03, "TT":1017.05, "EE":432.58, "TE":677.91, "TTTEEE":2124.51}


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
                "debug": True,
                "likelihood": {"hillik_spt.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params[mode], **fg_params[mode]},
                "packages_path": packages_path,
            }
            
            model = get_model(info)
            print( f"COBAYA/{mode}: {-2*model.loglikes({})[0][0]}")
            self.assertLess( abs(-2*model.loglikes({})[0][0] - chi2), 1)


if __name__ == "__main__":
    unittest.main()
