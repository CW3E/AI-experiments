# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
from typing import Optional
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

def generate_run_forward_jitted(model_config, task_config, params, state, 
                                diffs_stddev_by_level, mean_by_level, stddev_by_level):

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

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
    return run_forward_jitted

