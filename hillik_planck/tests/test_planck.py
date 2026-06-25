import os
import unittest

import numpy as np
from cobaya.install import resolve_packages_path
packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or resolve_packages_path()

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
        "PLK_Adust100TT": 27.,
        "PLK_Adust143TT": 21.,
        "PLK_Adust217TT": 10.,
        "Acib": 1.03,
        "Atsz": 6.,
        "Aksz": 1.,
        "xi": 0.1,
        "beta_cib": 1.75,
        "beta_radio": -0.8,
        "PLK_radio_TT": 60.,
        "PLK_cib_ps": 6.,
        "PLK_alpha_dustTT": -2.6
    },
    "EE": {
        "PLK_Adust100EE": 0.8,
        "PLK_Adust143EE": 0.3,
        "PLK_Adust217EE": 0.2,
    },
    "TE": {
        "PLK_Adust100TE": 1.6,
        "PLK_Adust143TE": 0.8,
        "PLK_Adust217TE": 0.4,
    },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}

#expected_chi2 = {"TT": 9231.9894, "EE": 9509.2059, "TE": 10214.672, "TTTEEE": 13138.09}
#expected_chi2 = {"TT": 4810.9264, "EE": 1805.0759, "TE": 1985.9408}
expected_chi2 = {
    "TT": 4810.9264,
    "TTTEEE": 8578.9722,
}

expected_dof = {
    "TT": 1646,
    "TTTEEE": 4872,
}

expected_lmax = {
    "TT": 2500,
    "TTTEEE": 2500,
}

expected_lmin = {
    "TT": 30,
    "TTTEEE": 30,
}

def minimum_lmin(likelihood):
    return min(
        min(likelihood._lmins[mode])
        for mode in ["TT", "TE", "EE"]
        if likelihood._is_mode[mode]
    )

class HillikPlkTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in expected_chi2:
            install(
                {"likelihood": {f"hillik_planck.{mode}": None}},
                path=packages_path,
                no_set_global=True,
            )
        print("\n" + "=" * 80)
        print("Starting hillik_planck regression checks")
        print("=" * 80)

##     def test_camb(self):
##         import camb
##         import hillik_planck

##         camb_cosmo = cosmo_params.copy()
##         camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
##         pars = camb.set_params(**camb_cosmo)
##         results = camb.get_results(pars)
##         powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##         cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

##         for mode, chi2 in expected_chi2.items():
##             _hlp = getattr(hillik_planck, mode)
##             my_lik = _hlp({"debug": True,"packages_path": packages_path})
##             loglike = my_lik.loglike(cl_dict, **{**calib_params, **nuisance_params[mode]})
##             print( f"CAMB/{mode}: {-2*loglike}")
##             self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for mode in expected_chi2:
            with self.subTest(mode=mode):
                likelihood_name = f"hillik_planck.{mode}"
                info = {
                    "debug": False,
                    "likelihood": {likelihood_name: None},
                    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                    "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                    "packages_path": packages_path,
                }
                model = get_model(info)
                measured_chi2 = -2 * model.loglikes({})[0][0]
                likelihood = model.likelihood[likelihood_name]

                print(f"{likelihood_name}:  {measured_chi2} (measured),  {expected_chi2[mode]} (expected),  diff={measured_chi2-expected_chi2[mode]}")
                self.assertAlmostEqual(measured_chi2, expected_chi2[mode], delta=1)
                self.assertEqual(minimum_lmin(likelihood), expected_lmin[mode])
                self.assertEqual(likelihood.lmax, expected_lmax[mode])
                self.assertEqual(likelihood.dof(), expected_dof[mode])
                self.assertEqual(likelihood._invkll.shape, (expected_dof[mode], expected_dof[mode]))
                self.assertTrue(np.array_equal(likelihood._invkll, likelihood._invkll.T))
                self.assertEqual(len(likelihood.delta_dl), expected_dof[mode])
