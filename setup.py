from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="hillik",
    version="0.0.1",
    description="A cobaya high-ell CMB likelihood",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matthieu Tristram",
    url="https://github.com/mtristram/hillik",
    license="GNU license",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["astropy", "cobaya>=3.0"],
    package_data={"hillik_planck": ["Hillipop.yaml", "Hillipop.bibtex"],
                  "hillik_spt": ["spt_hiell_2020.yaml","spt_hiell_2020.bibtex"],
                  "hillik_act": ["actpol_full_dr4.yaml","actpol_full_dr4.bibtex"],
                  },
)
