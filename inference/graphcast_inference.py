import os
os.chdir('/path/to/your/directory')

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

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_tree.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


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

#if global_rank == 0:
#    eval_targets.to_netcdf(dir_output+f'/ERA5_{startdate}_{enddate}.nc')  # start from 06, not 00

print("All Examples:  ", case_batch.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)


buffer = 10
obs_constraints = utils.set_constraint(eval_targets, constrains_domain, buffer)

## Autoregressive rollout

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

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
