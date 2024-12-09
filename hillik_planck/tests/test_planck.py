import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillik_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.010408,
    "As": 2.0956544e-09,
    "ombh2": 0.022223,
    "omch2": 0.119227,
    "ns": 0.9629,
    # "tau": 0.0563,
    "zrei": 8.0,
    "dz": 1.0,
}

calib_params = {
    "A_planck": 1.0,
    "PLK_cal_100A": 1.0,
    "PLK_cal_100B": 1.0,
    "PLK_cal_143A": 1.0,
    "PLK_cal_143B": 1.0,
    "PLK_cal_217A": 1.0,
    "PLK_cal_217B": 1.0,
    "PLK_pe_100A": 1.0,
    "PLK_pe_100B": 1.0,
    "PLK_pe_143A": 1.0,
    "PLK_pe_143B": 1.0,
    "PLK_pe_217A": 1.0,
    "PLK_pe_217B": 1.0,
    }

nuisance_params = {
    "TT": {
        "PLK_AdustT": 1.1,
        "Acib": 1.03,
        "Atsz": 6.,
        "Aksz": 1.,
        "xi": 0.1,
        "beta_cib": 1.75,
        "PLK_radio_ps": 60.,
        "PLK_cib_ps": 6.,
        },
    "EE": {
        "PLK_AdustP": 1.,
        },
    "TE": {
        "PLK_AdustT": 1.,
        "PLK_AdustP": 1.,
        },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}

chi2s = {"TT": 2057.94, "EE": 9506.97, "TE": 10101.83}


class HillikPlkTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"hillik_planck.{}".format(mode): None}},
                path=packages_path,
                skip_global=True,
                no_progress_bars=True,
            )

##     def test_camb(self):
##         import camb
##         import hillik_planck
        
##         camb_cosmo = cosmo_params.copy()
##         camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
##         pars = camb.set_params(**camb_cosmo)
##         results = camb.get_results(pars)
##         powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##         cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

##         for mode, chi2 in chi2s.items():
##             _hlp = getattr(hillik_planck, mode)
##             my_lik = _hlp({"debug": True,"packages_path": packages_path})
##             loglike = my_lik.loglike(cl_dict, **{**calib_params, **nuisance_params[mode]})
##             print( f"CAMB/{mode}: {-2*loglike}")
##             self.assertLess( abs(-2 * loglike - chi2), 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for mode, chi2 in chi2s.items():
            info = {
                "debug": False,
                "likelihood": {"hillik_planck.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}, "stop_at_error": True}},
                "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                "packages_path": packages_path,
            }

            model = get_model(info)
            # print(f"COBAYA/{mode}: {-2 * model.loglikes({})[0][0]}")
            self.assertLess(abs(-2 * model.loglikes({})[0][0] - chi2), 1)
