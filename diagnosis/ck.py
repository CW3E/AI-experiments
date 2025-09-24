import os
os.chdir('/path/to/your/directory')  
import numpy as np
import xarray as xr
from utils_array import sellonlat
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as mpconsts

dir_root = '/path/to/your/directory/'
dir_output = '/path/to/graphcast/output/'
dir_era5 = '/path/to/ERA5/'


def ck_kosaka(up:xr.DataArray, vp:xr.DataArray, um:xr.DataArray, vm:xr.DataArray, ifinteg=False):

    up = up.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')
    vp = vp.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')
    um = um.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')
    vm = vm.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')

    du_dx, du_dy = mpcalc.gradient(um, axes=[-1, -2])
    dv_dx, dv_dy = mpcalc.gradient(vm, axes=[-1, -2])
    du_dx = du_dx.values 
    du_dy = du_dy.values
    dv_dx = dv_dx.values
    dv_dy = dv_dy.values

    term1 = (vp**2 - up**2) * 0.5 * (du_dx - dv_dy)
    term2 = up * vp * (du_dy + dv_dx)
    ck = term1 - term2

    if ifinteg:
        return ck.integrate('level')/mpconsts.g*100
    else:
        return ck
    

def calculate_ck(u_raw, v_raw, startdate, domain):

    assert u_raw.time.identical(v_raw.time)
    dates = np.datetime64(f'{startdate}:00:00') + u_raw.time
    dates = dates.dt.strftime('%Y%m%dT%H').values
    hour_steps = {'00':0, '06':1, '12':2, '18':3}
    
    u_raw = u_raw.reindex(lat=u_raw.lat[::-1]) # 4D
    v_raw = v_raw.reindex(lat=v_raw.lat[::-1]) # 4D
    u_raw = af.sellonlat(u_raw, domain=domain, nst=1)
    v_raw = af.sellonlat(v_raw, domain=domain, nst=1)  
    ck = xr.full_like(u_raw.isel(level=0), np.nan).drop_vars('level')  # 3D

    for i, date in enumerate(dates):
        mmdd = date[4:8]
        hour = date[9:]
        print(mmdd, hour)

        u_clim = xr.open_dataset(dir_era5+f'ERA5.u.{mmdd}.clim.nc')['u'][hour_steps[hour]].loc[u_raw.level] # 3D
        u_clim = af.sellonlat(u_clim, domain=domain)
        v_clim = xr.open_dataset(dir_era5+f'ERA5.v.{mmdd}.clim.nc')['v'][hour_steps[hour]].loc[v_raw.level]  # 3D
        v_clim = af.sellonlat(v_clim, domain=domain) 

        # calculate UV anomaly
        u_anom = xr.full_like(u_clim, np.nan)  # 3D
        u_anom[:] = u_raw[i].values - u_clim.values
        
        v_anom = xr.full_like(v_clim, np.nan)
        v_anom[:] = v_raw[i].values - v_clim.values

        # calculate CK
        ck_temp = ck_kosaka(u_anom, v_anom, u_clim, v_clim, ifinteg=True, ifmetpy=True)  
        ck[i] = ck_temp.values

    return ck


startdate = '2016-03-05T00'
enddate = '2016-03-13T00'
lonlim, latlim = [-100, 65], [10, 90]

predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')

u_raw_free = predictions_free['u_component_of_wind'][:, 0]
v_raw_free = predictions_free['v_component_of_wind'][:, 0]
ck_free, u_climatology = calculate_ck(u_raw_free, v_raw_free, startdate, lonlim+latlim)

u_raw_constrained = predictions_constrained['u_component_of_wind'][:, 0]
v_raw_constrained = predictions_constrained['v_component_of_wind'][:, 0]
ck_constrained = calculate_ck(u_raw_constrained, v_raw_constrained, startdate, lonlim+latlim)

ck_free.to_netcdf(dir_output+f'CK_free_{startdate}_{enddate}.nc', mode='w')
ck_constrained.to_netcdf(dir_output+f'CK_constrained_{startdate}_{enddate}.nc', mode='w')
