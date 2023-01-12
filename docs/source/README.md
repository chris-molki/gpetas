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



from matplotlib import pyplot as plt
from jax import numpy as jnp
from gaussian_toolbox as gt

R, D = 10, 1
mu = jnp.zeros((R, D))
Sigma = jnp.ones((R, D, D))
Sigma = Sigma.at[:,0,0].set(jnp.linspace(.1,1,R))


p_X = gt.pdf.GaussianPDF(Sigma=Sigma, mu=mu)

x = jnp.linspace(-5,5,1000)[:,None]

plt.plot(x[:,0], p_X(x).T)
plt.xlabel('X')
plt.ylabel('p(X)')
plt.show()
```

For more details find the tutorials below.
