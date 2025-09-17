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
## North America heatwave
dataset_file = 'input.2021-06-14.18days.nc'
dataset_clim_file = 'clim.2021-06-14.18days.nc'
startdates = ['2021-06-21T00','2021-06-21T06','2021-06-21T12','2021-06-21T18','2021-06-22T00']   
enddates = [(str(np.datetime64(s) + np.timedelta64(8, 'D'))[:13]) for s in startdates]
constrains_domain = [100, 160, 15, 45] 

## Scandinavia blocking
# dataset_file = 'input.2016-03-02.14days.nc'
# dataset_clim_file = 'clim.2016-03-02.14days.nc'
# startdates = ['2016-03-05T00','2016-03-05T06','2016-03-05T12','2016-03-05T18','2016-03-06T00']   
# enddates = [(str(np.datetime64(s) + np.timedelta64(8, 'D'))[:13]) for s in startdates]
# constrains_domain = [285, 310, 25, 50] 
########################################

## In parallel
global_rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
startdate = startdates[global_rank]
enddate = enddates[global_rank] 

startdate_clim = f'1979{startdate[4:]}'
enddate_clim = f'1979{enddate[4:]}'

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
case_batch_clim = xr.load_dataset(dir_input+dataset_clim_file).compute()

## Extract data based on initial time and target time

case_batch = utils.select_period(case_batch, startdate, enddate)
case_batch_clim = utils.select_period(case_batch_clim, startdate_clim, enddate_clim)
assert case_batch["time"].size >= 3  # 2 for input, >=1 for targets


## Extract training and eval data

eval_steps = case_batch["time"].size - 2  

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    case_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

clim_inputs, clim_targets, _ = data_utils.extract_inputs_targets_forcings(
    case_batch_clim, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

assert eval_targets["time"].size == clim_targets["time"].size

# if global_rank == 0:
    # eval_targets.to_netcdf(dir_output+f'/ERA5_{startdate}_{enddate}.nc')

print("All Examples:  ", case_batch.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)


buffer = 10
clim_constraints = utils.set_constraint(clim_targets, constrains_domain, buffer)
clim_inputs = utils.set_constraint(clim_inputs, constrains_domain, buffer)
obs_constraints = utils.set_constraint(eval_targets, constrains_domain, buffer)


relaxation = 1.0
varnames = list(eval_targets.data_vars.keys())  # skip forcing variables in eval_inputs
print("Variables: ", varnames)
constraint_weights = utils.set_constraint_weights(clim_constraints['10m_u_component_of_wind'][0,0,...], buffer)
constraint_weights.drop_vars('time')
constraint_weights *= relaxation 
constraint_weights_inverse = 1 - constraint_weights

#%%
## Replace Obs with climatology for the input
eval_inputs_changed = eval_inputs.copy()
for varname in varnames:
    constraints_var = clim_inputs[varname] # [1,2,181,360]
    replaced = eval_inputs_changed[varname].loc[..., constraints_var.lat, constraints_var.lon].copy()
    # replaced['time'] = constraints_var.time
    prescribed = constraints_var*constraint_weights + replaced*constraint_weights_inverse
    # prescribed = utils.permute_dims(prescribed, 0, 1)  # predictions: [time, batch, ...]; prescribed: [batch, time, ...]
    writable_data = np.full_like(eval_inputs_changed[varname].values, np.nan)
    writable_data[...] = eval_inputs_changed[varname].values # copy.copy(predictions[varname].data)
    lon_indices = np.where(eval_inputs_changed.lon.isin(clim_constraints.lon).values)[0]
    lat_indices = np.where(eval_inputs_changed.lat.isin(clim_constraints.lat).values)[0]
    writable_data[..., lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1] = prescribed.values
    writable_data = xr.DataArray(writable_data, dims=eval_inputs_changed[varname].dims, coords=eval_inputs_changed[varname].coords)

    eval_inputs_changed[varname] = writable_data
##############################################################

eval_inputs.to_netcdf(dir_output+f'inputs_{startdate}.nc')
eval_inputs_changed.to_netcdf(dir_output+f'inputs_clim-constrained_{startdate}.nc')

#%%
## Autoregressive rollout

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

# predictions_free = rollout.chunked_prediction(
#     run_forward_jitted,
#     rng=jax.random.PRNGKey(0),
#     inputs=eval_inputs,
#     targets_template=eval_targets * np.nan,
#     forcings=eval_forcings)
# # print('Free forecast done!')

## TSC
predictions_constrained = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings, 
    constraints=obs_constraints,
    buffer=buffer,
    relaxation=relaxation, 
    )
predictions_constrained.attrs['constrains_domain'] = constrains_domain

## CC
predictions_clim_constrained = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs_changed, 
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings, 
    constraints=clim_constraints,
    buffer=buffer,
    relaxation=relaxation, 
    num_steps_for_constraints=20, 
    )
predictions_clim_constrained.attrs['constrains_domain'] = constrains_domain
# print('Clim-constrained forecast done!')

## Output
predictions_constrained.to_netcdf(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc') 
predictions_clim_constrained.to_netcdf(dir_output+f'predictions_clim-constrained_{startdate}_{enddate}_5days.nc') 
