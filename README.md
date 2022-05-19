HiLLik: High-L Likelihood for CMB
=================================
[![Unit test](https://img.shields.io/github/workflow/status/mtristram/hillik/Unit%20test)](https://github.com/mtristram/hillik/actions/workflows/testing.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


``Hillik`` is a multifrequency CMB likelihood for CMB data. The likelihood is a spectrum-based
Gaussian approximation for cross-correlation spectra from Planck, SPT and ACT.
It is based on different public codes:
- [Planck Hillipop](https://github.com/planck-npipe/hillipop)
- [SPTlik](https://github.com/xgarrido/spt_likelihoods)
- [ACTpol](https://github.com/mtristram/actpol_full_dr4)

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
* ``hillik_planck``, Planck 2020 (PR4) [Planck Collaboration 2020](https://arxiv.org/abs/2007.04997)
* ``hillik_spt``, SPT high-l [Reichardt et al. 2020](https://arxiv.org/abs/2002.06197)
* ``hillik_act``, ACT DR4 Baseline Multi-frquency Likelihood presented in [Choi et al. 2020](https://arxiv.org/abs/2007.07289)

Install
-------

It is better to create a working directory

```shell
$ mkdir HillikWork
$ cd HillikWork
$ export COBAYA_DIR=$PWD
$ mkdir software
$ mkdir modules
```

Optionnal: you can make a python virtual env (note that in that case, you need to source at every log)
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

The ``-e`` option allow the developer to make changes within the `Hillipop` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Data
----

Data for the likelihoods are installed automatically by cobaya. Just type
```shell
$ cobaya-install -p $COBAYA_DIR/modules your_file.yaml
```

For the foregrounds template models, directly untar the tarball:
```shell
$ tar zxvf software/hillik/data/foregrounds_template.tar.gz --directory=$COBAYA_DIR/modules
```

Test
----

Then to test `cobaya`
```shell
$ cobaya-run -p $COBAYA_DIR/modules software/hillik/example/hillik_plk.yaml
```


Requirements
------------
* Python >= 3.5
* `numpy`
* `astropy`
