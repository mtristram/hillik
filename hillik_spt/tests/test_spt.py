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
    # "tau": 0.060,
    "zrei": 8.0,
    "dz": 1.0,
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
        Apksz=0.,
        Apksz_derived={'derived': True},
        Ahksz_derived={'derived': True},
        Atsz_derived={'derived': True},
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

chi2s = {"TThighl": 623.51, "TT": 1024.11, "EE": 439.48, "TE": 678.79, "TTTEEE": 2138.48}

inifiles = ['test_sz_3tp.yaml', 'test_pksz_1rf.yaml', 'test_hksz_1rf.yaml', 'test_tsz_1rf.yaml']
chi2s_sz = [1482.36, 11137.52, 10847.21, 4922.36]


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
        from cobaya.yaml import yaml_load_file
        from cobaya.run import run

        for mode, chi2 in chi2s.items():
            # print(mode)
            info = {
                "debug": False,
                "likelihood": {"hillik_spt.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}, "stop_at_error": True}},
                "params": {**cosmo_params, **calib_params[mode], **fg_params[mode]},
                "packages_path": packages_path,
            }

            model = get_model(info)
            print(f"COBAYA/{mode}: {-2*model.loglikes({}, return_derived=False)[0]}")
            self.assertLess(abs(-2*model.loglikes({}, return_derived=False)[0] - chi2), 1)

            if mode == "TThighl":
                for chi2, inifile in zip(chi2s_sz, inifiles):
                    _, sampler = run(yaml_load_file('./'+inifile))
                    print(sampler.logposterior.loglike)
                    self.assertLess(abs(-2.*sampler.logposterior.loglike - chi2), 1.)


if __name__ == "__main__":
    unittest.main()
