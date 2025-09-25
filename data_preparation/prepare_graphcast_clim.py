import sys
sys.path.append('../utils/')
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import utils

###############################
## Scandinavia blocking
startdate = '2016-03-02'
nday = 14
center_date_str = "0308"  

## North America heatwave
# startdate = '2021-06-14'
# nday = 18
# center_date_str = "0622"  
###############################

dir_era5 = '/path/to/ERA5/'
levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
variables_pl = ['t', 'z', 'u', 'v', 'w', 'q']
variables_sl = ['u10', 'v10', 't2m', 'msl', 'tp', 'tisr']  
variables_slpl = variables_sl + variables_pl
variables_slpl = variables_slpl#[:1]  # for testing
enddate = utils.get_enddate(startdate, nday)
year = datetime.strptime(startdate, "%Y-%m-%d").year
mmdds = pd.date_range(startdate, enddate, freq='1D').strftime('%m%d')

ds_clim = []
for var in variables_slpl:
    print(var)
    ds_dates = []
    for mmdd in mmdds:
        print(mmdd)
        with xr.open_dataset(dir_era5+f'{var}/climatology/ERA5.{var}.{mmdd}.clim.nc') as ds:    
            if hasattr(ds, 'level'):
                ds_dates.append(ds.sel(level=levels))
            else:
                ds_dates.append(ds)
    ds_dates = xr.concat(ds_dates, dim='time')
    ds_clim.append(ds_dates)

dates_range = utils.get_days_range(center_date_str, year=year, days_range_half=30)  # datetime.datetime
mmdd_range = [date.strftime("%m%d") for date in dates_range]
dates_range = [date.strftime("%Y-%m-%d") for date in dates_range]
print(dates_range)

## seasonal mean in that year
year1 = int(dates_range[0][:4])
year2 = int(dates_range[-1][:4])
ds_mean = []  
if year1 == year2:
    print('One year:', year1)
    for var in variables_slpl:
        print(var)
        with xr.open_dataset(dir_era5+f'{var}/ERA5_{var}_{year}.nc') as ds:    
            if hasattr(ds, 'level'):
                ds = ds.sel(time=slice(dates_range[0], dates_range[-1]), level=levels)
            else:
                ds = ds.sel(time=slice(dates_range[0], dates_range[-1]))
            ds_mean.append(ds.mean('time'))  # 61-day mean
else:   
    print('Two years:', year1, year2)
    for var in variables_slpl:
        print(var)
        with xr.open_dataset(dir_era5+f'{var}/ERA5_{var}_{year1}.nc') as ds1:    
            if hasattr(ds1, 'level'):
                ds1 = ds1.sel(time=slice(dates_range[0], f'{year1}-12-31'), level=levels)
            else:
                ds1 = ds1.sel(time=slice(dates_range[0], f'{year1}-12-31'))
        with xr.open_dataset(dir_era5+f'{var}/ERA5_{var}_{year2}.nc') as ds2:    
            if hasattr(ds2, 'level'):
                ds2 = ds2.sel(time=slice(dates_range[0], f'{year1}-12-31'), level=levels)
            else:
                ds2 = ds2.sel(time=slice(dates_range[0], f'{year1}-12-31'))
        ds = xr.concat([ds1, ds2], dim='time')
        print(ds[var].shape)
        ds_mean.append(ds.mean('time'))  # 61-day mean


## seasonal mean of the corresponding calendar days in climatology
ds_mean_clim = []  
for var in variables_slpl:
    print(var)
    ds_dates = []
    for mmdd in mmdd_range:
        # print(mmdd)
        with xr.open_dataset(dir_era5+f'{var}/climatology/ERA5.{var}.{mmdd}.clim.nc') as ds:    
            if hasattr(ds, 'level'):
                ds_dates.append(ds.sel(level=levels))
            else:
                ds_dates.append(ds)
    ds_dates = xr.concat(ds_dates, dim='time')
    print(ds_dates[var].shape)
    ds_mean_clim.append(ds_dates.mean('time'))  # 61-day mean


for i, var in enumerate(variables_slpl):
    print(var)
    assert hasattr(ds_mean[i], var)  # in that year
    assert hasattr(ds_mean_clim[i], var)
    assert hasattr(ds_clim[i], var)
    season_diff = ds_mean[i] - ds_mean_clim[i]  # no time dimension
    ds_clim[i] += season_diff  # 3D + 2D (4D + 3D)

## Merge all variables
ds_clim = xr.merge(ds_clim)  
ds_clim = utils.change_coords(ds_clim)
ds_clim = utils.convert_lon(ds_clim)

with xr.open_dataset(dir_era5+'ERA5.invariants.nc') as static:   # 0-360
    print('static')
    static = static.squeeze().drop_vars('time')  
    static = utils.change_coords(static)

ds_clim = xr.merge([ds_clim, static])


# %%
varnames = list(ds_clim.data_vars)
ds_clim.to_netcdf(f'/path/to/graphcast/input/clim.{startdate}.{nday}days.nc',
                  encoding=dict(zip(varnames, [{'dtype': np.float32}] * len(varnames))))

