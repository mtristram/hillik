HiLLik: High-L Likelihood for CMB
=================================

[![Unit test](https://github.com/mtristram/hillik/actions/workflows/testing.yml/badge.svg)](https://github.com/mtristram/hillik/actions/workflows/testing.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


``Hillik`` is a multifrequency CMB likelihood for CMB data. The likelihood is a spectrum-based
Gaussian approximation for cross-correlation spectra from Planck, SPT and ACT.
It is based on different public codes:
- [Planck Hillipop](https://github.com/planck-npipe/hillipop)
- [CANDL](https://github.com/Lbalkenhol/candl_data)
- [ACT-DR6](https://github.com/ACTCollaboration/act_dr6_mflike)

The foreground model is coherent over the three datasets and includes several foregrounds residuals in spectra domain:
- Galactic dust;
- the cosmic infrared background;
- thermal Sunyaev-Zeldovich emission;
- kinetic Sunyaev-Zeldovich emission;
- a tSZ-CIB correlation consistent with both models above; and
- unresolved point sources as a Poisson-like power spectrum.

It is interfaced with the [``cobaya``](https://cobaya.readthedocs.io/en/latest/) MCMC sampler.

Likelihood versions
-------------------

Likelihoods available are:
* ``hillik_planck.[TT|TE|EE|TTTEEE]``, Planck 2020 (PR4) [Planck Collaboration 2020](https://arxiv.org/abs/2007.04997)
* ``hillik_act.[TT|TE|EE|TTTEEE]``, ACT (DR6) Baseline Multi-frquency Likelihood presented in [Louis et al. 2025](https://arxiv.org/abs/2503.14452)
* ``hillik_spt.[TT|TE|EE|TTTEEE]``, SPT3G (D1) [Camphuis et al. 2025](https://arxiv.org/abs/2506.20707)

Planck-ACT cut aliases
----------------------

For combined Planck+ACT analyses, Hillik provides matching cut aliases for the
``hillik_planck.[TT|TE|EE|TTTEEE]`` and
``hillik_act.[TT|TE|EE|TTTEEE]`` likelihoods. Use the same alias family on both
sides, for example:

```yaml
likelihood:
  hillik_planck.TTTEEE_tristram2026cut: null
  hillik_act.TTTEEE_tristram2026cut: null
```

Available alias families are:

| Alias family           | TT / TE / EE split | Meaning |
| ---                    | ---                | ---     |
| ``*_tristram2026cut``  | 2000 / 1500 / 1000 | Planck-ACT split used in the [Tristram et al. (2026)](https://arxiv.org/abs/2511.04733) analysis and in the combined example/parfiles. |
| ``*_minerrcut``        | 1820 / 1070 / 820  | Split at the Planck/ACT uncertainty crossover, using Planck at lower multipoles and ACT at higher multipoles. |
| ``*_PACTcut``          |  1000 / 600 / 600  | Original P-ACT split, constrained by the [original ACT DR6 baseline cuts](https://arxiv.org/abs/2503.14452) (includes Planck-ACT bin overlap). |
| ``*_PACTcut_0overlap`` |   975 / 575 / 575  | Non-overlapping version of ``*_PACTcut``; removes the highest Planck bins until Planck and ACT bin supports do not overlap. |

These aliases are implemented through the ``lrange`` option. Users can also
provide custom ``lrange`` values directly to the base likelihoods. ACT cuts are
applied using the full bandpower-window support, so the effective retained ACT
bin edge can be slightly above the nominal requested threshold.

Install
-------

It is better to create a working directory

```shell
$ mkdir HillikWork
$ cd HillikWork
$ export COBAYA_DIR=$PWD
$ export COBAYA_PACKAGES_PATH=$PWD/modules
$ mkdir software
$ mkdir modules
```
For more details, please refer to the installing web page from cobaya [here](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html#using-the-automatic-installer).

Optional: you can make a python virtual env (note that in that case, you need to source at every log)
```shell
$ python -m venv pyenv
$ source pyenv/bin/activate
$ pip install -U pip
$ pip install -U ipython
```

Then clone this ``hillik`` repository 

```shell
$ git clone https://github.com/mtristram/hillik.git software/hillik
```

Then you can install the `Hillik` likelihoods and its dependencies *via*

```shell
$ pip install -e software/hillik
```

The ``-e`` option allow the developer to make changes within the `Hillik` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Data
----

Data for the likelihoods are installed automatically by cobaya. Just type
```shell
$ cobaya-install -p $COBAYA_PACKAGES_PATH your_file.yaml
```

For the foregrounds template models, you need to untar manually the tarball:
```shell
$ ln -s $COBAYA_DIR/software/hillik/data $COBAYA_PACKAGES_PATH/data/foregrounds
```

Test
----

Then to test `cobaya`
```shell
$ cobaya-run -p $COBAYA_DIR/modules software/hillik/example/hillik_plk.yaml
```

Requirements
------------
* Python >= 3.8
* `numpy`
* `cobaya` >= 3.5
* `astropy` >= 6.0.1
* `sacc` >= 0.9.0

How to cite Hillik
------------------
If you use this likelihood, please cite the following paper:
> Combining cosmic microwave background datasets with consistent foreground modelling\
> M. Tristram, et al., A&A, [2511.04733](https://arxiv.org/abs/2511.04733)\
> DOI: [https://doi.org/10.1051/0004-6361/202558015](https://doi.org/10.1051/0004-6361/202558015)
