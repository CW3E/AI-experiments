import os
os.chdir('/path/to/your/directory')
import sys
sys.path.append('../utils/')
import dataclasses
import functools
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray as xr
import utils 
from runner_utils import generate_run_forward_jitted

########################################
## Scandinavia blocking
dataset_file = 'input.2016-03-02.14days.nc'
hour = "00"
startdates = [f'2016-03-03T{hour}', f'2016-03-04T{hour}', f'2016-03-05T{hour}', f'2016-03-06T{hour}', f'2016-03-07T{hour}', 
              f'2016-03-08T{hour}', f'2016-03-09T{hour}', f'2016-03-10T{hour}', f'2016-03-11T{hour}', f'2016-03-12T{hour}']
enddate = '2016-03-13T00'
constrains_domain = [285, 310, 25, 50] 

## North America heatwave
# dataset_file = 'input.2021-06-14.18days.nc'
# hour = "00"
# startdates = [f'2021-06-19T{hour}', f'2021-06-20T{hour}', f'2021-06-21T{hour}', f'2021-06-22T{hour}', f'2021-06-23T{hour}', 
#               f'2021-06-24T{hour}', f'2021-06-25T{hour}', f'2021-06-26T{hour}', f'2021-06-27T{hour}', f'2021-06-28T{hour}']
# enddate = '2021-06-29T00'
# constrains_domain = [100, 160, 15, 45] 
########################################

## In parallel
global_rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
startdate = startdates[global_rank]

## Paths
dir_ckpt = '/path/to/graphcast/params/'
dir_stats = '/path/to/graphcast/stats/'
dir_input = '/path/to/graphcast/input/'
dir_output = '/path/to/graphcast/output/'

## Load the Data and initialize the model
params_file = 'graphcast_params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
ckpt = checkpoint.load(dir_ckpt+params_file, graphcast.CheckPoint)
params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config

diffs_stddev_by_level = xr.load_dataset(dir_stats+'stats_diffs_stddev_by_level.nc').compute()
mean_by_level = xr.load_dataset(dir_stats+'stats_mean_by_level.nc').compute()
stddev_by_level = xr.load_dataset(dir_stats+'stats_stddev_by_level.nc').compute()

run_forward_jitted = generate_run_forward_jitted(model_config, task_config, params, state, 
                                                 diffs_stddev_by_level, mean_by_level, stddev_by_level)

## Load data
case_batch = xr.load_dataset(dir_input+dataset_file).compute()

## Extract data based on initial time and target time
case_batch = utils.select_period(case_batch, startdate, enddate)
assert case_batch["time"].size >= 3  # 2 for input, >=1 for targets

## Extract training and eval data
eval_steps = case_batch["time"].size - 2  # 4 * 11 #12

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    case_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))


buffer = 10
obs_constraints = utils.set_constraint(eval_targets, constrains_domain, buffer)

## Autoregressive rollout
predictions_free = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)

predictions_constrained = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings, 
    constraints=obs_constraints,
    buffer=buffer,
    relaxation=1.0, 
    )
predictions_constrained.attrs['constrains_domain'] = constrains_domain

## Output
predictions_free.to_netcdf(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained.to_netcdf(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
