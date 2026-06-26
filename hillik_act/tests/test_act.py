import os
import unittest

import numpy as np
from cobaya.install import resolve_packages_path
packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or resolve_packages_path()

cosmo_params = {
    "cosmomc_theta": 0.01040,
    "As": 2.1178e-09,
    "ombh2": 0.022591,
    "omch2": 0.1238,
    "ns": 0.9666,
    "tau": 0.056,
    "Alens": 1.0,
}

ACTmaps = ["dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150"]

calib_params = {
    "ACT_cal": 1.0,
    **{f"ACT_cal_{m}": 1.0 for m in ACTmaps},
    **{f"ACT_pe_{m}": 1.0 for m in ACTmaps},
    **{f"ACT_band_shift_{m}": 0.0 for m in ACTmaps}
}

nuisance_params = {
    "TT": {
        'ACT_AdustTT': 7.97,
        'ACT_beta_dustTT': 1.5,
        'ACT_alpha_dustTT': -2.6,
        "Acib": 3.69,
        "Atsz": 3.35,
        "Aksz": 1.48,
        "xi": 0.088,
        "beta_cib": 1.87,
        "beta_radio": -0.78,
        "beta_dusty": 1.87,
        "ACT_cib_ps": 7.65,
        "ACT_radio_TT": 2.86,
    },
    "TE": {
        'ACT_AdustTE': 0.42,
        'ACT_beta_dustTE': 1.5,
        'ACT_alpha_dustTE': -2.4,
    },
    "EE": {
        'ACT_AdustEE': 0.17,
        'ACT_beta_dustEE': 1.5,
        'ACT_alpha_dustEE': -2.4,
        'ACT_radio_EE': 0.0,
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
        "likelihood": {"likelihood_name": "hillik_act.TT"},
        "chi2": 3254.41,
        "dof":  601,
        "lrange": {"TT": (576, 7925)},
    },
    "EE": {
        "likelihood": {"likelihood_name": "hillik_act.EE"},
        "chi2": 979.42,
        "dof": 406,
        "lrange": {"EE": (576, 7925)},
    },
    "TE": {
        "likelihood": {"likelihood_name": "hillik_act.TE"},
        "chi2": 1925.22,
        "dof":  644,
        "lrange": {"TE": (576, 7925), "ET": (776, 7925)},
    },
    "TTTEEE": {
        "likelihood": {"likelihood_name": "hillik_act.TTTEEE"},
        "chi2": 6133.82,
        "dof": 1651,
        "lrange": {"TT": (576, 7925), "TE": (576, 7925), "ET": (776, 7925), "EE": (576, 7925)},
    },
    "TTTEEE_planckcut": {
        "likelihood": {"likelihood_name": "hillik_act.TTTEEE_planckcut"},
        "chi2": 3712.30,
        "dof": 1139,
        "lrange": {"TT": (2026, 7925), "TE": (1526, 7925), "ET": (1526, 7925), "EE": (1026, 7925)},
    },
}

def measured_lranges(likelihood):
    ranges = {}
    for spec in likelihood.spectra:
        for mode in spec["polarizations"]:
            bpw = spec[mode]["bpw"]
            if bpw.weight.shape[0] == len(bpw.values):
                support = np.any(bpw.weight != 0, axis=1)
            else:
                support = np.any(bpw.weight != 0, axis=0)
            lmin = min(bpw.values[support])
            lmax = max(bpw.values[support])
            if mode not in ranges:
                ranges[mode] = (lmin, lmax)
            else:
                ranges[mode] = (
                    min(ranges[mode][0], lmin),
                    max(ranges[mode][1], lmax),
                )
    return ranges

class HillikACTTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install(
            {"likelihood": {"hillik_act.TTTEEE": None}},
            path=packages_path,
            no_set_global=True,
        )
        print("\n" + "=" * 80)
        print("Starting hillik_act regression checks")
        print("=" * 80 + "\n")

##     def test_camb(self):
##         import camb
##         import hillik_act

##         camb_cosmo = cosmo_params.copy()
##         camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
##         pars = camb.set_params(**camb_cosmo)
##         results = camb.get_results(pars)
##         powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##         dl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

##         for mode, chi2 in expected_chi2.items():
##             _act = getattr(hillik_act, mode)({"packages_path": packages_path})
##             loglike = _act.loglike(dl_dict, **{**calib_params,**nuisance_params[mode]})
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
                self.assertEqual(likelihood.inv_cov.shape, (expected["dof"], expected["dof"]))
                self.assertEqual(len(likelihood.delta_dl), expected["dof"])


if __name__ == "__main__":
    unittest.main()
