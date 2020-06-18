"""
AutoBFE
"""

import setuptools
from os import path

root = path.dirname(__file__)

with open(path.join(root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(root, "requirements_Auto_BFE.txt")) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="AutoBFE",
    version="0.1",
    url="https://github.com/LBNL-ETA/AutoBFE",
    download_url="https://github.com/LBNL-ETA/AutoBFE",
    license="MIT",
    maintainer="samirtouzani",
    maintainer_email="samirtouzani@lbl.gov",
    description="Toolbox for satellite/aerial image feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
