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
        "Adust_PLK_100T": 2.43,
        "Adust_PLK_143T": 3.06,
        "Adust_PLK_217T": 9.03,
        "Acib": 3.2,
        "Atsz": 9.3,
        "Aksz": 11.55,
        "xi": 2.93,
        "beta_cib": 1.49,
        "Aps_PLK_100x100": 354.,
        "Aps_PLK_100x143": 168.,
        "Aps_PLK_100x217": 186.,
        "Aps_PLK_143x143": 100.,
        "Aps_PLK_143x217": 108.,
        "Aps_PLK_217x217": 122.,
        },
    "EE": {
        "Adust_PLK_100P": 0.19,
        "Adust_PLK_143P": 0.47,
        "Adust_PLK_217P": 0.99,
        },
    "TE": {
        "Adust_PLK_100T": 0.50,
        "Adust_PLK_143T": 0.64,
        "Adust_PLK_217T": 1.23,
        "Adust_PLK_100P": 0.09,
        "Adust_PLK_143P": 1.27,
        "Adust_PLK_217P": 3.80,
        },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}

chi2s = {"TT": 13327.19, "EE": 9224.61, "TE": 9899.60} #, 'TTTEEE':33167.95}
#chi2s = {"TT": 11636.29}
#chi2s = {"TT": 10472.62, 'EE':9413.73, 'TE':10079.78}#, 'TTTEEE':30772.38}  #ell<2000


class HillikPlkTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"hillik_planck.{}".format(mode): None}},
                path=packages_path,
                skip_global=True,
            )

    def test_hillipop(self):
        import camb
        import hillik_planck
        
        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

        for mode, chi2 in chi2s.items():
            _hlp = getattr(hillik_planck, mode)
            my_lik = _hlp({"debug": True,"packages_path": packages_path})
            loglike = my_lik.loglike(cl_dict, **{**calib_params, **nuisance_params[mode]})
            self.assertLess( abs(-2 * loglike - chi2), 1)

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
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)
