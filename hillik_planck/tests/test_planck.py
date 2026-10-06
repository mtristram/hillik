import os
import unittest
import numpy as np
import camb
from astropy.utils import minversion

from cobaya.install import resolve_packages_path
packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or resolve_packages_path()

CAMB2 = minversion(camb, "2.0.0")

cosmo_params = {
    "cosmomc_theta": 0.010408,
    "As": 2.0956544e-09,
    "ombh2": 0.022223,
    "omch2": 0.119227,
    "ns": 0.9629,
    "tau": 0.0563,
}

PLKmaps = ["100A", "100B", "143A", "143B", "217A", "217B"]

calib_params = {
    "A_planck": 1.0,
    **{f"PLK_cal_{m}": 1.0 for m in PLKmaps},
    **{f"PLK_pe_{m}": 1.0 for m in PLKmaps},
}

nuisance_params = {
    "TT": {
        "PLK_Adust100TT": 27.,
        "PLK_Adust143TT": 21.,
        "PLK_Adust217TT": 10.,
        "PLK_alpha_dustTT": -2.6,
        "Acib": 1.03,
        "Atsz": 6.,
        "Aksz": 1.,
        "xi": 0.1,
        "beta_cib": 1.75,
        "beta_radio": -0.8,
        "PLK_cib_ps": 6.,
        "PLK_radio_TT": 60.,
    },
    "TE": {
        "PLK_Adust100TE": 1.6,
        "PLK_Adust143TE": 0.8,
        "PLK_Adust217TE": 0.4,
    },
    "EE": {
        "PLK_Adust100EE": 0.8,
        "PLK_Adust143EE": 0.3,
        "PLK_Adust217EE": 0.2,
    },
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
    **nuisance_params["EE"],
}


# Test cases and their expectation values
expectations = {
    "TT": {
        "likelihood": {"likelihood_name": "hillik_planck.TT"},
        "chi2": 4802.5479 if CAMB2 else 4810.9264,
        "dof": 1646,
        "lrange": {"TT": (30, 2500)},
    },
    "TTTEEE__cut_TT_only": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 2500]}},
        "chi2": 4802.5479if CAMB2 else 4810.9264,
        "dof": 1646,
        "lrange": {"TT": (30, 2500)},
    },
    "TTTEEE": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE"},
        "chi2": 8569.0671 if CAMB2 else 8578.9722,
        "dof": 4872,
        "lrange": {"TT": (30, 2500), "TE": (30, 2000), "EE": (30, 2000)},
    },
    "TTTEEE_tristram2026cut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE_tristram2026cut"},
        "chi2": 7382.0091 if CAMB2 else 7392.6049,
        "dof": 4202,
        "lrange": {"TT": (30, 2000), "TE": (30, 1500), "EE": (30, 1000)},
    },
    "TTTEEE__cut_tristram2026cut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 2005], "TE": [30, 1505], "EE": [30, 1000]}},
        "chi2": 7382.0091 if CAMB2 else 7392.6049,
        "dof": 4202,
        "lrange": {"TT": (30, 2000), "TE": (30, 1500), "EE": (30, 1000)},
    },
    "TTTEEE_minerrcut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE_minerrcut"},
        "chi2": 6531.3537 if CAMB2 else 6540.027,
        "dof": 3782,
        "lrange": {"TT": (30, 1820), "TE": (30, 1070), "EE": (30, 820)},
    },
    "TTTEEE__cut_minerrcut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 1820], "TE": [30, 1070], "EE": [30, 820]}},
        "chi2": 6531.3537 if CAMB2 else 6540.027,
        "dof": 3782,
        "lrange": {"TT": (30, 1820), "TE": (30, 1070), "EE": (30, 820)},
    },
    "TTTEEE_PACTcut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE_PACTcut"},
        "chi2": 3752.7005 if CAMB2 else 3753.8726,
        "dof": 2972,
        "lrange": {"TT": (30, 1000), "TE": (30, 600), "EE": (30, 600)},
    },
    "TTTEEE__cut_PACTcut": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 1000], "TE": [30, 600], "EE": [30, 600]}},
        "chi2": 3752.7005 if CAMB2 else 3753.8726,
        "dof": 2972,
        "lrange": {"TT": (30, 1000), "TE": (30, 600), "EE": (30, 600)},
    },
    "TTTEEE_PACTcut_0overlap": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE_PACTcut_0overlap"},
        "chi2": 3620.4946 if CAMB2 else 3621.5991,
        "dof": 2918,
        "lrange": {"TT": (30, 970), "TE": (30, 570), "EE": (30, 570)},
    },
    "TTTEEE__cut_PACTcut_0overlap": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 975], "TE": [30, 575], "EE": [30, 575]}},
        "chi2": 3620.4946 if CAMB2 else 3621.5991,
        "dof": 2918,
        "lrange": {"TT": (30, 970), "TE": (30, 570), "EE": (30, 570)},
    },
    "TTTEEE__cut_TT_1000": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"TT": [30, 1000]}},
        "chi2": 1581.4566 if CAMB2 else 1581.7426,
        "dof": 1096,
        "lrange": {"TT": (30, 1000)},
    },
    "TTTEEE__cut_EE_1000": {
        "likelihood": {"likelihood_name": "hillik_planck.TTTEEE", "lrange": {"EE": [30, 1000]}},
        "chi2":  1363.5381 if CAMB2 else 1364.6279,
        "dof": 1046,
        "lrange": {"EE": (30, 1000)},
    },
}

def measured_lranges(likelihood):
    return {
        mode: (min(likelihood._lmins[mode]), max(likelihood._lmaxs[mode]))
        for mode in ["TT", "TE", "EE"]
        if likelihood._is_mode[mode]
    }

class HillikPLKTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for expected in expectations.values():
            install(
                {"likelihood": {expected["likelihood"]["likelihood_name"]: None}},
                path=packages_path,
                no_set_global=True,
            )
        print("\n" + "=" * 80)
        print("Starting hillik_planck regression checks")
        print("=" * 80 + "\n")

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

        for label, expected in expectations.items():
            with self.subTest(label=label):
                likelihood = expected["likelihood"].copy()
                likelihood_name = likelihood.pop("likelihood_name")
                mode = likelihood_name.split(".")[-1].split("_", 1)[0]
                info = {
                    "debug": False,
                    "likelihood": {likelihood_name: likelihood or None},
                    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                    "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                    "packages_path": packages_path,
                }
                model = get_model(info)
                measured_chi2 = -2 * model.loglikes({})[0][0]
                likelihood = model.likelihood[likelihood_name]

                print(f"{label}:  {measured_chi2} (measured),  {expected['chi2']} (expected),  diff={measured_chi2-expected['chi2']}\n")
                self.assertAlmostEqual(measured_chi2, expected["chi2"], delta=1)
                self.assertEqual(measured_lranges(likelihood), expected["lrange"])
                self.assertEqual(likelihood.dof(), expected["dof"])
                self.assertEqual(likelihood._invkll.shape, (expected["dof"], expected["dof"]))
                self.assertTrue(np.array_equal(likelihood._invkll, likelihood._invkll.T))
                self.assertEqual(len(likelihood.delta_dl), expected["dof"])
