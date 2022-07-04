Templates for HiLLik
====================

The foreground model in ``Hillik`` is coherent over the three datasets and includes several foregrounds residuals in spectra domain:
- Galactic dust;
- the cosmic infrared background;
- thermal Sunyaev-Zeldovich emission;
- kinetic Sunyaev-Zeldovich emission;
- a tSZ-CIB correlation consistent with both models above; and
- unresolved point sources as a Poisson-like power spectrum.

Template should be provided in Dl = l(l+1)/2pi Cl, for every multipole
up to 13500 (otherwise filled by zeros).
Units are muK^2.
The file should be a txt file with two columns: ell, dl
