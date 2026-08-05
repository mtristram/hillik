import os
import unittest

import numpy as np
from cobaya.install import resolve_packages_path
packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or resolve_packages_path()

cosmo_params = {
    "cosmomc_theta": 0.01040,
    "As": 2.16185225e-09,
    "ombh2": 0.02241,
    "omch2": 0.1188,
    "ns": 0.9686,
    "tau": 0.060,
    "Alens": 1.0,
}

SPTmaps = [90, 150, 220]

calib_params = {
    "SPT3G_cal": 1.0,
    **{f"SPT3G_cal_{m}": 1.0 for m in SPTmaps},
    **{f"SPT3G_pe_{m}": 1.0 for m in SPTmaps},
    **{f"SPT3G_T2P2_{fq}": 0. for fq in SPTmaps},
    **{f"SPT3G_beta_{i+1}": -0.5 for i in range(9)},
    **{f"SPT3G_beta_pol_{fq}": 0.5 for fq in SPTmaps},
    "SPT3G_kappa": 5e-6,
}

nuisance_params = {
    "TT": {
        "SPT3G_AdustTT": 1.98,
        "SPT3G_beta_dustTT": 1.5,
        "SPT3G_alpha_dustTT": -2.53,
        "Acib": 3.0,
        "Atsz": 0.94,
        "Aksz": 2.3,
        "xi": 0.26,
        "beta_cib": 1.80,
        "beta_radio": -0.8,
        "beta_dusty": 1.80,
        "T_cib": 25.,
        "SPT3G_cib_ps": 8.,
        "SPT3G_radio_TT": 1.,
#           "SPT3G_ps_90x90":  10.7, "SPT3G_ps_90x150":  8.6, "SPT3G_ps_90x220": 16.6,
#           "SPT3G_ps_150x150":11.9, "SPT3G_ps_150x220":32.0, "SPT3G_ps_220x220":95.0,
    },
    "TE": {
       "SPT3G_AdustTE": 0.10,
       "SPT3G_beta_dustTE": 1.5,
       "SPT3G_alpha_dustTE": -2.40,
    },
    "EE": {
       "SPT3G_AdustEE": 0.05,
       "SPT3G_beta_dustEE": 1.5,
       "SPT3G_alpha_dustEE": -2.40,
       "SPT3G_radio_EE": 0.0,
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
        "likelihood": {"likelihood_name": "hillik_spt.TT"},
        "chi2": 897.167,
        "dof": 312,
        "lrange": {"TT": (350, 4095)},
    },
    "EE": {
        "likelihood": {"likelihood_name": "hillik_spt.EE"},
        "chi2": 612.755,
        "dof": 432,
        "lrange": {"EE": (350, 4095)},
    },
    "TE": {
        "likelihood": {"likelihood_name": "hillik_spt.TE"},
        "chi2": 827.760,
        "dof": 648,
        "lrange": {"TE": (350, 4095)},
    },
    "TTTEEE": {
        "likelihood": {"likelihood_name": "hillik_spt.TTTEEE"},
        "chi2": 2300.714,
        "dof": 1392,
        "lrange": {"TT": (350, 4095), "TE": (350, 4095), "EE": (350, 4095)},
    },
}

def measured_lranges(likelihood):
    ranges = {}
    for i, spec in enumerate(likelihood.spectra_to_fit):
        mode = spec[:2]
        selected_windows = likelihood.windows[spec][likelihood.spec_bin_min[i]-1:likelihood.spec_bin_max[i]]
        support = np.any(selected_windows != 0, axis=0)
        supported_ells = likelihood.lmin + np.flatnonzero(support)
        if mode not in ranges:
            ranges[mode] = (supported_ells[0], supported_ells[-1])
        else:
            ranges[mode] = (
                min(ranges[mode][0], supported_ells[0]),
                max(ranges[mode][1], supported_ells[-1]),
            )
    return ranges

class HillikSPTTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install(
            {"likelihood": {"hillik_spt.TTTEEE": None}},
            path=packages_path,
            no_set_global=True,
        )
        print("\n" + "=" * 80)
        print("Starting hillik_spt regression checks")
        print("=" * 80 + "\n")

##     def test_camb(self):
##         import camb
##         import hillik_spt

##         camb_cosmo = cosmo_params.copy()
##         for mode, chi2 in expected_chi2.items():
##             _spt = getattr(hillik_spt, mode)({"debug":True,"packages_path":packages_path})

##             camb_cosmo.update({"lmax": 10000, "lens_potential_accuracy": 1})
##             pars = camb.set_params(**camb_cosmo)
##             results = camb.get_results(pars)
##             powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
##             cl_boltz = {k: powers["total"][:, v] for k, v in {"tt": 0, "TT": 0, "EE": 1, "TE": 3}.items()}

##             loglike = _spt.loglike(cl_boltz, **nuisance_params[mode],**calib_params[mode])
##             print( f"CAMB/{mode}: {-2*loglike}")
##             self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        from cobaya.model import get_model

        for label, expected in expectations.items():
            with self.subTest(label=label):
                likelihood = expected["likelihood"].copy()
                likelihood_name = likelihood.pop("likelihood_name")
                info = {
                    "debug": False,
                    "likelihood": {likelihood_name: likelihood or None},
                    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                    "params": {**cosmo_params, **calib_params, **nuisance_params['TTTEEE']},
                    "packages_path": packages_path,
                }
                model = get_model(info)
                measured_chi2 = -2 * model.loglikes({})[0][0]
                likelihood = model.likelihood[likelihood_name]

                print(f"{label}:  {measured_chi2} (measured),  {expected['chi2']} (expected),  diff={measured_chi2-expected['chi2']}\n")
                self.assertAlmostEqual(measured_chi2, expected["chi2"], delta=1)
                self.assertEqual(measured_lranges(likelihood), expected["lrange"])
                self.assertEqual(likelihood.dof(), expected["dof"])
                self.assertEqual(likelihood._inv_bpcov.shape, (expected["dof"], expected["dof"]))
                self.assertEqual(len(likelihood.delta_dl), expected["dof"])


if __name__ == "__main__":
    unittest.main()
