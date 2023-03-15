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
    "tau": 0.0563,
}

calib_params = {
    "A_planck": 1.0,
    "cal_PLK": 1.0,
    "cal_PLK_100A": 0.0,
    "cal_PLK_100B": 0.0,
    "cal_PLK_143A": 0.0,
    "cal_PLK_143B": 0.0,
    "cal_PLK_217A": 0.0,
    "cal_PLK_217B": 0.0,
    "calE_PLK_100A": 0.0,
    "calE_PLK_100B": 0.0,
    "calE_PLK_143A": 0.0,
    "calE_PLK_143B": 0.0,
    "calE_PLK_217A": 0.0,
    "calE_PLK_217B": 0.0,
    }

nuisance_params = {
    "TT": {
        "Adust_PLK_100T": 1.37,
        "Adust_PLK_143T": 1.85,
        "Adust_PLK_217T": 8.48,
        "Acib": 1.26,
        "Atsz": 2.65,
        "Aksz": 15.66,
        "xi": 7.,
        "beta_cib": 1.77,
        "Aps_PLK_100x100": 390.37,
        "Aps_PLK_100x143": 179.36,
        "Aps_PLK_100x217": 177.54,
        "Aps_PLK_143x143": 102.30,
        "Aps_PLK_143x217": 112.19,
        "Aps_PLK_217x217": 134.16,
        },
    "EE": {
        "Adust_PLK_100P": 0.18,
        "Adust_PLK_143P": 0.46,
        "Adust_PLK_217P": 1.15,
        },
    "TE": {
        "Adust_PLK_100T": 1.22,
        "Adust_PLK_143T": 1.89,
        "Adust_PLK_217T": 8.16,
        "Adust_PLK_100P": 0.18,
        "Adust_PLK_143P": 0.45,
        "Adust_PLK_217P": 0.66,
        },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}

chi2s = {"TT": 11261.91, "EE": 9236.68, "TE": 9921.18}
chi2s = {"TT": 11261.91}


class HillikPlkTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"hillik_planck.{}".format(mode): None}},
                path=packages_path,
                skip_global=True,
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
                "debug": True,
                "likelihood": {"hillik_planck.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                "packages_path": packages_path,
            }
            
            model = get_model(info)
#            print( f"COBAYA/{mode}: {-2 * model.loglikes({})[0][0]}")
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)
