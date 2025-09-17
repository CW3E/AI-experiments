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

def q1_yanai(t, u, v, omega, dt=None, intg=False):

    u = u.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')
    v = v.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('m/s')
    t = t.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units.kelvin
    omega = omega.metpy.assign_crs(grid_mapping_name='latitude_longitude') * units('Pa/s')
    if hasattr(t, 'level'):
        p = t.level*100*units.Pa
    else:
        raise ValueError('No level named level!')

    ##----horizontal temperature advection
    t_adv = mpcalc.advection(t, u=u, v=v)  # including minus sign

    if hasattr(t, 'time'):
        ss = mpcalc.static_stability(p, t, vertical_dim=-3).transpose(*t.dims)*p/mpconsts.Rd
    else:
        ss = mpcalc.static_stability(p, t, vertical_dim=-3)*p/mpconsts.Rd
    ss = mpcalc.smooth_n_point(ss, 9)

    ##----total
    if dt is None:
        q1 = - mpconsts.Cp_d * (t_adv + omega*ss) 
    else:
        ##----local dT/dt (K/s)
        dTdt = t.differentiate(coord='time', datetime_unit='s') / units('s') #* units('kelvin/s')
        q1 = mpconsts.Cp_d * (dTdt - t_adv - omega*ss)

    if intg:
        new_dims = tuple([i for i in t.dims if i != 'level'])
        return q1.integrate('level').transpose(*new_dims)/mpconsts.g*100
    else:
        return q1
    

def calculate_q1(u_raw, v_raw, t_raw, w_raw, domain):
    # lonw, lone, lats, latn = domain  
    
    t_raw = t_raw.reindex(lat=t_raw.lat[::-1]) # 3D
    u_raw = u_raw.reindex(lat=u_raw.lat[::-1])
    v_raw = v_raw.reindex(lat=v_raw.lat[::-1])
    w_raw = w_raw.reindex(lat=w_raw.lat[::-1])
    q1 = q1_yanai(t_raw, u_raw, v_raw, w_raw, dt=21600, intg=True)

    q1 = af.sellonlat(q1, domain=domain, nst=1) 
    q1_noboundary = xr.full_like(q1, np.nan)
    q1_noboundary[:, 1:-1, 1:-1] = q1[:, 1:-1, 1:-1]
    q1_noboundary.name = 'q1'

    return q1_noboundary.drop_vars('metpy_crs')


startdate = '2021-06-21T00'
enddate = '2021-06-29T00'
lonlim, latlim = [90, 270], [0, 80]
step_start = 0
step_end = -1 

predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')


u_raw = predictions_free['u_component_of_wind'][:, 0].loc[:, 100:] # 3 time steps to calculate Q1 (dT/dt)
v_raw = predictions_free['v_component_of_wind'][:, 0].loc[:, 100:]
t_raw = predictions_free['temperature'][:, 0].loc[:, 100:]
w_raw = predictions_free['vertical_velocity'][:, 0].loc[:, 100:]
q1_free = calculate_q1(u_raw, v_raw, t_raw, w_raw, lonlim+latlim)

u_raw = predictions_constrained['u_component_of_wind'][:, 0].loc[:, 100:] # 3 time steps to calculate Q1 (dT/dt)
v_raw = predictions_constrained['v_component_of_wind'][:, 0].loc[:, 100:]
t_raw = predictions_constrained['temperature'][:, 0].loc[:, 100:]
w_raw = predictions_constrained['vertical_velocity'][:, 0].loc[:, 100:]
q1_constrained = calculate_q1(u_raw, v_raw, t_raw, w_raw, lonlim+latlim)

q1_time = np.datetime64(f'{startdate}:00:00') + q1_free.time

q1_free.to_netcdf(dir_output+f'Q1_free_{startdate}_{enddate}.nc', mode='w')
q1_constrained.to_netcdf(dir_output+f'Q1_constrained_{startdate}_{enddate}.nc', mode='w')
