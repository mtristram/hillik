import os
import unittest

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
    **{f"ACT_band_shift_{m}": 0. for m in ACTmaps}
}

extgal_params = {
    "Acib": 3.69,
    "Atsz": 3.35,
    "Aksz": 1.48,
    "xi": 0.088,
    "beta_cib": 1.87,
    "beta_radio": -0.78,
    "beta_dusty": 1.87,
    "ACT_cib_ps": 7.65,
    "ACT_radio_TT": 2.86,
}

fg_params = {
    'TT':{'ACT_AdustTT': 7.97, 'ACT_beta_dustTT':1.5, 'ACT_alpha_dustTT':-2.6},
    'TE':{'ACT_AdustTE': 0.42, 'ACT_beta_dustTE':1.5, 'ACT_alpha_dustTE':-2.4},
    'EE':{'ACT_AdustEE': 0.17, 'ACT_beta_dustEE':1.5, 'ACT_alpha_dustEE':-2.4, 'ACT_radio_EE': 0.0},
}

nuisance_params = {
    "TT": {**fg_params['TT'],**extgal_params},
    "EE": {**fg_params['EE']},
    "TE": {**fg_params['TE']},
    "TTTEEE": {**fg_params['TT'],**extgal_params,**fg_params['TE'],**fg_params['EE']},
    }
nuisance_params["TTTEEE_PACT"] = nuisance_params["TTTEEE"]

expected_chi2 = {
    "TT": 3254.41,
    "EE":  979.42,
    "TE": 1925.22,
    "TTTEEE": 6133.82,
    "TTTEEE_PACT": 3712.30,
}

expected_dof = {
    "TT": 601,
    "EE": 406,
    "TE": 644,
    "TTTEEE": 1651,
    "TTTEEE_PACT": 1139,
}

expected_lmax = {
    "TT": 8501,
    "EE": 8501,
    "TE": 8501,
    "TTTEEE": 8501,
    "TTTEEE_PACT": 8501,
}

expected_lmin = {
    "TT": 600,
    "EE": 600,
    "TE": 600,
    "TTTEEE": 600,
    "TTTEEE_PACT": 1000,
}

def minimum_lmin(likelihood):
    return min(
        spec["scales"][mode][0]
        for spec in likelihood.spectra
        for mode in spec["polarizations"]
    )

class ACTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in expected_chi2:
            install(
                {"likelihood": {f"hillik_act.{mode}": None}},
                path=packages_path,
                no_set_global=True,
            )
        print("\n" + "=" * 80)
        print("Starting hillik_act regression checks")
        print("=" * 80)

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

        for mode in expected_chi2:
            with self.subTest(mode=mode):
                likelihood_name = f"hillik_act.{mode}"
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
                self.assertEqual(likelihood.inv_cov.shape, (expected_dof[mode], expected_dof[mode]))
                self.assertEqual(len(likelihood.delta_dl), expected_dof[mode])


if __name__ == "__main__":
    unittest.main()
