import xarray as xr
import numpy as np
import utils

dir_in = '/path/to/graphcast/input/'
date = '2016-03-02' 
nday = 14

pres_levels = xr.open_dataset(dir_in+f'ERA5.{date}.{nday}days.pressure_levels.nc')
single_level = xr.open_dataset(dir_in+f'ERA5.{date}.{nday}days.single_level.nc')
tp = xr.open_dataset(dir_in+f'ERA5.{date}.{nday}days.total_precipitation.nc').rename({'valid_time': 'time'})
static = xr.open_dataset(dir_in+f'ERA5.invariants.nc')

tp_6h = tp['tp'].rolling(time=6).sum()
tp_6h = tp_6h.loc[date:]
tp_6h = tp_6h[tp_6h.time.dt.hour.isin([0, 6, 12, 18])]

static = static.squeeze().drop_vars('time')  # remove time dimension

pres_levels = utils.change_coords(pres_levels)
single_level = utils.change_coords(single_level)
tp_6h = utils.change_coords(tp_6h)
tp_6h = xr.Dataset({'total_precipitation_6hr':tp_6h})
static = utils.change_coords(static)
# datetime = xr.Dataset({'datetime':pres_levels['datetime']}) 
# datetime = change_coords(datetime)

inputs = xr.merge([single_level, pres_levels, tp_6h, static])

if hasattr(inputs, 'expver'):
    inputs = inputs.drop_vars("expver")
if hasattr(inputs, 'number'):
    inputs = inputs.drop_vars("number")

varnames  = list(inputs.data_vars)
inputs.to_netcdf(dir_in+f'input.{date}.{nday}days.nc', 
                 encoding=dict(zip(varnames, [{'dtype': np.float32}] * len(varnames)))
                 )