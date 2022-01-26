HiLLik: High-L Likelihood for CMB
=================================
[![Unit test](https://img.shields.io/github/workflow/status/mtristram/hillik/Unit%20test)](https://github.com/mtristram/hillik/actions/workflows/testing.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


``Hillik`` is a multifrequency CMB likelihood for CMB data. The likelihood is a spectrum-based
Gaussian approximation for cross-correlation spectra from Planck, SPT and ACT.
It is based on different public codes:
- [Planck Hillipop](https://github.com/planck-npipe/hillipop)
- [SPTlik](https://github.com/xgarrido/spt_likelihoods)

The foreground model is coherent over the three datasets and includes several foregrounds residuals in spectra domain:
- Galactic dust;
- the cosmic infrared background;
- thermal Sunyaev-Zeldovich emission;
- kinetic Sunyaev-Zeldovich emission;
- a tSZ-CIB correlation consistent with both models above; and
- unresolved point sources as a Poisson-like power spectrum.

Likelihoods available are ``hillik.planck``, ``hillik.spt``, ``hillik.act``.

It is interfaced with the [https://cobaya.readthedocs.io/en/latest/](``cobaya``) MCMC sampler.

Likelihood versions
-------------------

* ``planck_2020_hillipop`` Planck 2020 (PR4)
* ``spt_hiell_2020`` SPT high-l `Reichardt et al. <https://arxiv.org/abs/2002.06197>`_, 2020

Install
-------
The easiest way to install the `Hillipop` likelihood is *via* `pip`

```shell
$ pip install planck-2020-hillipop [--user]
```

If you plan to dig into the code, it is better to clone this repository to some location

```shell
$ git clone https://github.com/planck-npipe/hillipop.git /where/to/clone
```

Then you can install the `Hillipop` likelihood and its dependencies *via*

```shell
$ pip install -e /where/to/clone
```

The ``-e`` option allow the developer to make changes within the `Hillipop` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Installing Hillipop likelihood data
-----------------------------------

The [`examples/hillipop_example.yaml`](examples/hillipop_example.yaml) file is a good starting point to
know the different nuisance parameters used by `hillipop` likelihoods.

You should use the `cobaya-install` binary to automatically download the data needed by the
`Hillipop` likelihood

```shell
$ cobaya-install /where/to/clone/examples/hillipop_example.yaml -p /where/to/put/packages
```

Data and code such as [CAMB](https://github.com/cmbant/CAMB) will be downloaded and installed within
the ``/where/to/put/packages`` directory. For more details, you can have a look to `cobaya`
[documentation](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html).

Requirements
------------
* Python >= 3.5
* `numpy`
* `astropy`
