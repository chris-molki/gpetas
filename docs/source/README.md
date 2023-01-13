# Install

## Create virtual environment
Just use the following commands using conda.
It creates virtual environment with python==3.9 installed and activates it
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
***Warning:*** Currently there is an issue with ```pandoc```, 
which is for example used when notebooks are embedded in the documentation.
It is not properly install via ```pip```, therefore
installed ```pandoc``` via ```conda```
```bash
pip uninstall pandoc
conda install pandoc
```
`gpetas` is installed in developer/editable mode.

## Build documentation locally

Change into the folder *docs*
and generate a local documentation in html.
```
cd docs
make html
```
For opening the documentation just open *index.html* file in a browser, e.g.,
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

```python
import gpetas
```
``gpetas`` is based on following main objects (implemented as python ``class``):
* ***data_obj*** has all information about the data and includes the ***domain_obj***
* ***domain_obj*** has all information about the temporal and spatial domain, time origin
* ***setup_obj*** has all information required for the inference procedures
* inference_obj (as ***GS_obj*** or ***mle_obj***)
  


For more details find the tutorials below.

## Provide data and generate a ``gpetas`` data object
Provide data and save it as a `gpetas` data 
object (python class), which is the basis for the inference.

### Define domain
You have to specify following aspects of 
the domain:
* time domain
  * time borders for the training *T_borders_training*
  * time borders for the testing *T_borders_testing*
  * time origin (when was '0.' days)
* spatial domain *X_borders*
* domain of the marks (magnitudes) usually by *m0*

```python
import gpetas

```
### Data from a text file
Observed data can be provided either directly 
in a text file with a specific format which ``gpetas`` can read.

* a text file with 5 columns, no header, where each row corresponds to a single event (earthquake)

    > idx, x_lon, y_lat, mag, time 

where *time* is decimal in days. Assuming N observed events, the data file has dimension (N,5), e.g.,
```
    1	2.7523	2.1859	4.6	0.000000
    2	2.7447	2.0783	3.4	0.004436
    3	3.0142	2.1734	3.4	0.320997
    4	3.4610	1.5132	3.5	2.271304
    5	2.8646	2.2552	3.4	11.108739
    6	2.8682	3.0405	4.1	41.841438
    7	2.8948	3.0755	4.9	41.963901
    8	2.6674	3.0231	3.4	42.026355
```
Generate ```gpetas``` data class
```python
import gpetas
fname = '<path to data file>' # as a string
data_obj = gpetas.some_fun.create_data_obj_from_cat_file(fname)
```
```data_obj``` is a class object and contains the data and all information about the data and the considered temporal domain, spatial domain and mark domain.
It is the bases for the inference.
See this [notebook](notebooks/01_getdata_from_file.ipynb)

#### Define domain


### Data from an online source
Data can be easily downloaded from online sources as
* USGS: HOST = 'earthquake.usgs.gov'

Here we use functionalities provided by the ``pycsep`` package which 
facilitates the data access.
Related functions require a 
*datetime.datetime*
to specify the start and end dates.
``pycsep`` provides access to the 
ComCat web API and to the 
Bollettino Sismico Italiano API using
* csep.query_comcat()
* csep.query_bsi().

In ``pycsep`` data is downloaded saved into 
a *catalog object* (pyhton class) which 
can be easily converted into a ``gpetas``` 
*data_obj* using ***data_obj__from_catalog_obj()***.

First install ``pycsep`` using ``conda``
```bash
conda activate gpetas_env
conda install --channel conda-forge pycsep
```
Start ``python`` and do the following.
```python
```






## Model setup

## Inference

### Bayesian inference
### Maximum likelihood (classical way)

## Results

## Prediction


