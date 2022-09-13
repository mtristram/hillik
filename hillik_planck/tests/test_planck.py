import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillik_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.010408,
    "As": 2.096925e-09,
    "ombh2": 0.022234,
    "omch2": 0.119176,
    "ns": 0.9632,
    "Alens": 1.0,
    "tau": 0.0564,
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
}

nuisance_params = {
    "TT": {
        "Adust_PLK_100T": 1.6,
        "Adust_PLK_143T": 2.1,
        "Adust_PLK_217T": 8.5,
        "Acib": 4.3,
        "Atsz": 9.3,
        "Aksz": 11.3,
        "xi": 1.8,
        "beta_cib": 1.3,
        "Aps_PLK_100x100": 360.,
        "Aps_PLK_100x143": 159.,
        "Aps_PLK_100x217": 156.,
        "Aps_PLK_143x143": 88.,
        "Aps_PLK_143x217": 90.,
        "Aps_PLK_217x217": 125.,
        },
    "EE": {
        "Adust_PLK_100P": 0.15,
        "Adust_PLK_143P": 0.3,
        "Adust_PLK_217P": 1.1,
        },
    "TE": {
        "Adust_PLK_100T": 1.6,
        "Adust_PLK_143T": 2.1,
        "Adust_PLK_217T": 8.5,
        "Adust_PLK_100P": 0.15,
        "Adust_PLK_143P": 0.3,
        "Adust_PLK_217P": 1.1,
        },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}

#chi2s = {"TT": 11415.58, "EE": 9244.86, "TE": 9916.65}
#chi2s = {"TT": 11636.29}
chi2s = {"TT": 10472.62}  #ell<2000
chi2s = {"TT": 10472.62, 'EE':9413.73, 'TE':10079.78}#, 'TTTEEE':30772.38}


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
            my_lik = _hlp({"packages_path": packages_path})
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
