import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillik_packages"
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
        "Adust_PLK_100": 0.6,
        "Adust_PLK_143": 1.24,
        "Adust_PLK_217": 7.9,
        "Acib": 3.2,
        "Atsz": 2.3,
        "Aksz": 5.5,
        "xi": 5.6,
        "beta_cib": 1.5,
        "Aps_PLK_100x100": 428.,
        "Aps_PLK_100x143": 211.,
        "Aps_PLK_100x217": 200.,
        "Aps_PLK_143x143": 130.,
        "Aps_PLK_143x217": 128.,
        "Aps_PLK_217x217": 151.,
        },
    "EE": {
        "Ad100P": 0.016,
        "Ad143P": 0.035,
        "Ad217P": 0.130,
        },
    "TE": {
        "Ad100T": 0.02,
        "Ad143T": 0.04,
        "Ad217T": 0.13,
        "Ad100P": 0.016,
        "Ad143P": 0.035,
        "Ad217P": 0.130,
        },
}
nuisance_params["TTTE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TTTE"],
    **nuisance_params["EE"],
}

#chi2s = {"TT": 11415.58, "EE": 9244.86, "TE": 9916.65}
chi2s = {"TT": 11636.29}


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
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {"hillik_planck.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)
