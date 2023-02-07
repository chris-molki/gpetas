# gpetas (GP-ETAS)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://chris-molki.github.io/gpetas)
![GitHub](https://img.shields.io/github/license/chris-molki/gpetas)

The main motivation of this package is to make
`gpetas` usable for a broader community.
The code follows this [Statistics and Computing paper](https://link.springer.com/article/10.1007/s11222-022-10085-3).

[**Basic Usage**](#basics) | [**Install guide**](#installation) | [**Citing**](#citation) | [**Documentation**](https://chris-molki.github.io/gpetas/)

# Basics

## ETAS model a Hawkes process
Spatio-temporal Hawkes process with conditional intensity function:

```math
\lambda(t,\boldsymbol{x}|H_t,\boldsymbol{\theta}_\mu,\boldsymbol{\theta}_\varphi) = \mu(\boldsymbol{x}|\boldsymbol{\theta}_\mu) + \sum_{i: t_i < t}\varphi(t-t_i,\boldsymbol{x}-\boldsymbol{x}_i|H_t,\boldsymbol{\theta}_\varphi).
```

## Bayesian non-parametric background intensity via Gaussian process modelling

```math
\lambda(t,\boldsymbol{x}|H_t,\boldsymbol{\theta}_\mu,\boldsymbol{\theta}_\varphi) = \frac{\bar{\lambda} }{1+e^{-f(\boldsymbol x)}}+ \sum_{i: t_i < t}\varphi(t-t_i,\boldsymbol x-\boldsymbol x_i|H_t,\boldsymbol \theta_\varphi).
```
* $f\sim GP()$

## Bayesian inference via data augmentation and Gibbs sampling

Posterior distribution has no simple closed form. We use Markov chain Monte Carlo (MCMC) methods for generating samples from 
the joint posterior distribution. More specifically we employ MH within Gibbs sampling.

# Installation

`gpetas` requires `python>=3.9`.
For a proper installation create a virtual environment first and activate it. Just do the following.

## Create virtual environment
```
conda create --name gpetas_env python=3.9
conda activate gpetas_env
```
This creates a virtual environment with python==3.9 installed, and activates it.
In order to return to the base environment afterwards, use: *conda deactivate*.

## Package installation
Clone the repository into a directory and change into the folder and do the following.
```
git clone https://github.com/chris-molki/gpetas.git
cd gpetas
pip install -r requirements.txt
pip install -r requirements_docs.txt
pip install -e .
```

`gpetas` has been installed in developer/editable mode.

***Hint:*** In case there is a problem with pip installation 
of a subpackage/library just try,
```
conda install -c conda-forge --file requirements.txt
```

### For online data downloads install pycsep
First make clear that *gpetas_env* is activated,
otherwise activate it via *conda activate gpetas_env*.
If *gpetas_env* is activate, install pycsep as follows,
```
conda install --channel conda-forge pycsep
```

## Build documentation locally

Change into the folder *docs*
and generate a local documentation in html.
```
cd docs
make html
```
For opening the documentation just open *index.html* file
```
open ./build/html/index.html 
```

## Uninstall ```gpetas```

Uninstall or remove ```gpetas``` and 
virtual *environment gpetas_env* easily 
as follows.
Just change to the parent directory 
and remove folder *gpetas* recursively
```
cd ..
rm -rf gpetas
```
Return to bases environment by deactivating virtual environment *gpetas_env*
```
conda deactivate
```
Subsequently remove virtual environment *gpetas_env* using the following 
command,
```
conda env remove -n gpetas_env
```
or just delete the directory of *gpetas_env*.

# Short installation guide
Clone the repository into a directory and go into the folder. Just do the following
```bash
pip install git+https://github.com/chris-molki/gpetas.git
```
Make sure the requirements (requirements.txt) are full-filled.

For code development do
```bash
git clone https://github.com/chris-molki/gpetas.git
cd gpetas/
pip install -r requirements.txt
pip install -e .
```

# Citation

To cite this repository:

```
@software{gpetas2023github,
	author = {CM,CD},
	title = {{gpetas}: A Python package for Bayesian inference of GP-ETAS model.},
	url = {https://github.com/chris-molki/gpetas},
	version = {0.0.1},
	year = {2023},
}
```
The development of the repository is based on the following
publication,

```
@article{molkenthin2022gp,
	title={GP-ETAS: semiparametric Bayesian inference for the spatio-temporal epidemic type
		aftershock sequence model},
	author={Molkenthin, Christian and Donner, Christian and Reich, Sebastian and Z{\"o}ller, Gert
		and Hainzl, Sebastian and Holschneider, Matthias and Opper, Manfred},
	journal={Statistics and Computing},
	volume={32},
	number={2},
	pages={1--25},
	year={2022},
	publisher={Springer}
}
```