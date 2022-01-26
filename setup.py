from setuptools import find_packages, setup

import versioneer

with open("README.md") as f:
    long_description = f.read()

setup(
    name="hillik",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    package_data={"planck_2020_hillipop": ["Hillipop.yaml", "Hillipop.bibtex"],
                  "spt_hiell_2020": ["spt_hiell_2020.yaml","spt_hiell_2020.bibtex"]},
)
