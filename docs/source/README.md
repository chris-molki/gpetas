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
* ***data_obj*** has all information about the data $\mathcal{D}=\\{(t_i,\boldsymbol{x}_i,m_i)\\}\_{i=1}\^{N\_{\mathcal{D}}}$ and includes the ***domain_obj***
* ***domain_obj*** has all information about
  * temporal domain $\mathcal{T_{\rm all}}\subset \mathcal{R}$ including absolute time origin, training window $\mathcal{T}$ and testing 
  window $\mathcal{T^\ast}$ with $\mathcal{T_{\rm all}}=\mathcal{T}\cup\mathcal{T^\ast}$.
  * spatial domain $\mathcal{X}\subset \mathcal{R}^2$
  * domain of the marks (earthquake magnitudes) implemented as $m\in[m_0,\infty)$
* ***setup_obj*** has all information required for the inference procedures, e.g., priors
but also the data (stored in *data_obj*), thus *setup_obj* includes 
*data_obj* which includes *domain_obj*: *domain_obj* $\subset$ *data_obj* $\subset$ *setup_obj*
* inference_obj (as ***GS_obj*** or ***mle_obj***) which is used to perform inference 
and storing the results. In particular, results of  ***GS_obj***, e.g. samples of the 
joint posterior among other things are saved in a ***save_GS_obj*** as saving 
***GS_obj*** is to heavy. In contrast, ***mle_obj*** can be directly saved and reloaded as 
it is much lighter in terms of storage.

For more details find the tutorials below.

## Provide data and generate a ``gpetas`` data object
Provide data and save it as a `gpetas` data 
object (python class), which is the basis for the inference.

### Define domain
You have to specify following aspects of 
the domain:
* time domain in days
  * time borders for the training *T_borders_training*
  * time borders for the testing *T_borders_testing*
  * time origin as *sting* (what corresponds to '0.' days) given by a *datetime* format: '%Y-%m-%d %H:%M:%S.%f',
  e.g.,'2010-01-01 00:00:00.0'
* spatial domain *X_borders* usually in degrees (Lon,Lat)
* domain of the marks (magnitudes) usually defined through *m0* as $m\in[m_0,\infty)$

```python
import gpetas
import numpy as np

# specify domain
time_origin = '2010-01-01 00:00:00.0'
T_borders_all = np.array([0.,4383.]) # until '2022-01-01 00:00:00.0'
T_borders_training = np.array([0.,3000.])
X_borders = np.array([[-120., -113.],[  30.,   37.]])
m0=3.5

# save specifications into domain_obj
domain_obj = gpetas.utils.R00x_setup.region_class()
domain_obj.T_borders_all = T_borders_all
domain_obj.T_borders_training=T_borders_training
domain_obj.T_borders_testing = np.array([T_borders_training[1],T_borders_all[1]])
domain_obj.time_origin = time_origin
domain_obj.X_borders = X_borders
domain_obj.m0 = m0
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
# automatic domain if domain is None
data_obj = gpetas.some_fun.create_data_obj_from_cat_file(fname=fname,domain_obj=domain_obj)
```
```data_obj``` is a python *class* and contains the data and all information as the data and the considered temporal domain, spatial domain and mark domain.
It is the bases for the inference.
See this [notebook](notebooks/01_getdata_from_file.ipynb)



### Data from an online source
Data can be easily downloaded from online sources as
* USGS: HOST = 'earthquake.usgs.gov'

Here we use functionalities provided by the ``pycsep`` package which 
facilitates the data access.
Related functions require a 
*datetime.datetime* object
to specify the start and end times/dates.
``pycsep`` provides access to the 
ComCat web API and to the 
Bollettino Sismico Italiano API using
* csep.query_comcat()
* csep.query_bsi().

In ``pycsep`` data is downloaded and stored into 
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
import gpetas
import numpy as np
import datetime
time_format = "%Y-%m-%d %H:%M:%S.%f"


import csep
from csep.utils import time_utils, comcat
```
Specify the domain of the data to be downloaded and save it to a *domain_obj* as
described before.

#### Californian data
```python
# time domain
time_origin = '2010-01-01 00:00:00.0'
time_end = '2022-01-01 00:00:00.0'
time_origin_obj = datetime.datetime.strptime(time_origin, time_format).replace(
                tzinfo=datetime.timezone.utc)
time_end_obj = datetime.datetime.strptime(time_end, time_format).replace(
                tzinfo=datetime.timezone.utc)
delta_Tall=(time_end_obj-time_origin_obj).total_seconds()/(60.*60.*24)
T_borders_all = np.array([0.,delta_Tall])
T_borders_training = np.array([0.,3000.])

# spatial domain
X_borders = np.array([[-120., -113.],[  30.,   37.]])

# mark domain: [m0,+inf)
m0=3.5

domain_obj = gpetas.utils.R00x_setup.region_class()
domain_obj.T_borders_all = T_borders_all
domain_obj.T_borders_training=T_borders_training
domain_obj.T_borders_testing = np.array([T_borders_training[1],T_borders_all[1]])
domain_obj.time_origin = time_origin
domain_obj.X_borders = X_borders
domain_obj.m0 = m0
vars(domain_obj)
```
Based on *domain_obj* use ``pycsep`` for 
downloading the data into a *catalog_obj*
```python
# get pycsep catalog object
start_time = time_origin_obj
end_time = time_end_obj
min_magnitude=domain_obj.m0
min_latitude=domain_obj.X_borders[1,0]
max_latitude=domain_obj.X_borders[1,1]
min_longitude=domain_obj.X_borders[0,0]
max_longitude=domain_obj.X_borders[0,1]
catalog_obj = csep.query_comcat(start_time=start_time, end_time=end_time, 
                        min_magnitude=min_magnitude, 
                        min_latitude=min_latitude,max_latitude=max_latitude, 
                        min_longitude=min_longitude, max_longitude=max_longitude)
```
Use ```gpetas``` routine *data_obj__from_catalog_obj()* to convert 
*catalog_obj* into a ```gpetas``` 
*data_obj*.
```python
# cat2data_obj
data_obj = gpetas.utils.get_data_pycsep.data_obj__from_catalog_obj(catalog_obj=catalog_obj,R_obj=domain_obj)
```
Data setup including domain information can 
be easily plotted as follows
```python
h=gpetas.plotting.plot_setting(data_obj=data_obj)
```
See this [notebook](notebooks/02_getdata_from_online_via_pycsep.ipynb)

## Inference setup, model setup
In order to perform inference (Bayesian or Maximum Likelihood estimation,(MLE))
one needs to define or setup several auxiliary variables, which for example
includes priors, number of samples or in the case of MLE stopping rules for 
iterative schemes etc.
This is done by creating a ***setup_obj*** for the inference which 
includes all required information of the
* Bayesian inference, i.e., Gibbs sampling procedure with Gaussian process 
modelling of the background intensity (GP-ETAS, gpetas) or
* classical Maximum Likelihood estimation (MLE) using a kernel density estimator 
for the background intensity

### Setup object for Bayesian inference (Gibbs sampler, gpetas)
In GP-ETAS (gpetas) we need to define priors on ...
In addition, we have to choose control parameters of the Gibbs sampler ...
A summary of the setup of GP-ETAS (gpetas) is given in the paper 
at page 13.

All this information is contained in the ```gpetas``` *setup_obj* which 
can be created as follows,
```python
import gpetas
import numpy as np
```
Load or generate *data_obj* as it is a central part of *setup_obj*.
```python
### load data_obj
case_name = 'Rxxx'
output_dir = './output/inference_results'
fname = output_dir+'/data_obj_%s.all'%case_name
data_obj = np.load(fname,allow_pickle=True)
```
Set variables for the Bayesian inference.
```python
### variables of the Gibbs sampler

# sampler parameters
burnin = 50                                # number of discared initial samples. default: 5000
Ksamples = 10                              # number of samples of the joint posterior default: 500 (a few hundreds)
thinning = 20                              # default:10 # or 20:thinning of the obtained samples in order to avoid autocorrelation
num_iterations = Ksamples*thinning+1
MH_proposals_offspring = 100               # Number of MH proposals for offspring params
MH_cov_empirical_yes = None                # using empirical cov for proposal distribution
sigma_proposal_offspring_params = None     # uses default values: 0.01**2 # alternatives:0.03**2
kth_sample_obj = None                      # starting sampling from kth sample 
print('#iters',num_iterations)


# offspring
prior_theta_dist = 'gamma'                 # specifies prior distribution either 'gamma' or 'uniform'
prior_theta_params = None
theta_start_Kcpadgq = None                 # uses default values:
spatial_offspring = 'R'                    # alternatives: 'G' gaussian 
stable_theta_sampling = 'yes'              # constraint on theta that only stable Hawkes processes are allowed


# bg: 
cov_params = None                          # start values of hypers, uses default: silverman rule
mu_nu0 = None                              # mean of hyper prior on nu_0, uses default value:



# bg: spatial resolution for plotting/evaluations
ngrid_per_dim = 50                         # default value: 50
X_grid = gpetas.some_fun.make_X_grid(data_obj.domain.X_borders, nbins=ngrid_per_dim)
                                           # generates spatial grid for plotting etc.
    
# general 
time_origin = data_obj.domain.time_origin
case_name = data_obj.case_name
    
# save results
outdir = output_dir
```
Create *setup_obj* and save it to a directory.
```python
setup_obj = gpetas.setup_Gibbs_sampler.setup_sampler(data_obj=data_obj,
             utm_yes=None,
             spatial_offspring=spatial_offspring,
             theta_start_Kcpadgq=theta_start_Kcpadgq,
             sigma_proposal_offspring_params=sigma_proposal_offspring_params,
             ngrid_per_dim=ngrid_per_dim,
             cov_params=cov_params,
             mu_nu0=None,
             X_grid=X_grid,
             outdir=outdir,
             prior_theta_dist=prior_theta_dist,
             prior_theta_params=prior_theta_params,
             stable_theta_sampling=stable_theta_sampling,
             time_origin=time_origin,
             case_name=case_name,
             burnin=burnin, 
             Ksamples=Ksamples,
             num_iterations=num_iterations,
             thinning=thinning,
             MH_proposals_offspring=MH_proposals_offspring,
             MH_cov_empirical_yes=MH_cov_empirical_yes,
             kth_sample_obj=kth_sample_obj)
```
This object is the bases for the Bayesian inference, 
the GP-ETAS Gibbs sampler. It contains *data_obj*, *domain_obj* and 
all required information in order to perform the inference.
See this [notebook](notebooks/03_inference_setup.ipynb)

For the sampling procedure and obtain posterior samples see
section [Bayesian inference via sampling](bayesian-inference-via-sampling).

### Setup object for classical MLE


## Inference

### Bayesian inference via sampling

### Maximum likelihood (classical way)

## Results

## Prediction


