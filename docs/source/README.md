# Install

Just use the following commands using conda.

## Create virtual environment
Creates virtual environment with python==3.9 installed and activates it
```
conda create --name gpetas_env python=3.9
conda activate gpetas_env
```
In order to return to the base environment afterwards, use: *conda deactivate*.

## Package installation
```
git clone https://github.com/chris-molki/gpetas.git
cd gpetas
pip install -r requirements.txt
pip install -r requirements_docs.txt
pip install -e .
```

`gpetas` is installed in developer/editable mode.

For uninstall just change to the parent directory *cd ..* and (1) remove gpetas directory *rm -rf gpetas* and (2) remove the virtual environment
using *env remove -n gpetas_env* (deactivate gpetas_env before as mentioned above).

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

## How to uninstall ```gpetas```

Uninstall or remove ```gpetas``` and
virtual environment *gpetas_env* easily
as follows.
Just change to the parent directory
and remove folder *gpetas*.
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
env remove -n gpetas_env
```
or just delete the directory of *gpetas_env*.

# Getting started

Then you can already get started:

```
import gpetas
```

For more details find the tutorials below.

## Provide data
Provide (i) observed data
or (ii) simulate data using, e.g., ``gpetas``.

### Observed data
Data are time series and have to have 
a specific format which ``gpetas`` can read.







## Model setup

## Inference

### Bayesian inference
### Maximum likelihood (classical way)

## Results

## Prediction


