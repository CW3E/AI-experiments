import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

varnames = {
    'u10':'10m_u_component_of_wind',
    'v10':'10m_v_component_of_wind',
    't2m':'2m_temperature',
    'msl':'mean_sea_level_pressure',
    'tisr':'toa_incident_solar_radiation',
    'z':'geopotential',
    't':'temperature',
    'u':'u_component_of_wind',
    'v':'v_component_of_wind',
    'w':'vertical_velocity',
    'q':'specific_humidity',
    'tp':'total_precipitation_6hr',
}

staticnames = {
        'lsm':'land_sea_mask',
        'z':'geopotential_at_surface',
        }

def get_enddate(startdate, length):
    # startdate: 'yyyy-mm-dd'
    enddate = pd.to_datetime(startdate) + pd.Timedelta(days=length-1)
    return enddate.strftime('%Y-%m-%d')

def change_coords(ds):
    
    if hasattr(ds, 'valid_time'):
        ds = ds.rename({'valid_time': 'time'})
    if hasattr(ds, 'pressure_level'):
        ds = ds.rename({'pressure_level': 'level'})
    
    if hasattr(ds, 'time'):
        ds['datetime'] = ds['time']
        hours = ds.time - ds.time.min()
        ds['time'] = hours
        ds = ds.expand_dims(dim='batch', axis=0)
        # ds = ds.assign_coords(datetime=('time', datetime.values))
        ds = ds.assign_coords(datetime=ds['datetime'])
        names = varnames
    else:
        names = staticnames
        
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.reindex(lat=ds['lat'][::-1])
    
    for key, val in names.items():
        if key in ds:
            ds = ds.rename({key: val})
    
    return ds

def convert_lon(ds):
    # Convert longitude from -180-180 to 0-360 
    lon = ds['lon'].values
    lon[lon < 0] += 360
    ds['lon'] = lon
    ds = ds.sortby('lon') 
    return ds


def get_days_range(center_date_str, year, days_range_half=30):
    try:
        center_date = datetime.strptime(f"{year}{center_date_str}", "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {center_date_str}, please input mmdd format.")

    start_date = center_date - timedelta(days=days_range_half)
    end_date = center_date + timedelta(days=days_range_half)

    # Generate a list of 91 dates in mmdd format
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)  # .strftime("%m%d")
        current_date += timedelta(days=1)

    assert len(date_list) == days_range_half * 2 + 1

    return date_list


def permute_dims(arr:xr.DataArray, d1, d2):
    # d1: old dims; d2: new dims
    if (type(d1) is int) and (type(d2) is int):
        dims = list(arr.dims)
        dims[d1], dims[d2] = dims[d2], dims[d1]
    elif (type(d1) is list) and (type(d2) is list):
        dims = np.array(arr.dims)
        d1, d2 = np.array(d1), np.array(d2)
        dims[d1] = dims[d2]
        dims = list(dims)
    else:
        raise ValueError('Wrong input')

    return arr.transpose(*dims)


## Extract data based on initial time and target time
def select_period(ds, startdate:str, enddate:str):
    dates = ds['datetime'].values
    startdate = np.datetime64(startdate)
    startdate_6hbefore = startdate - np.timedelta64(6, 'h')
    enddate = np.datetime64(enddate)
    mask = (dates >= startdate_6hbefore) & (dates <= enddate)
    mask = mask[0]
    time_subset = ds['time'].values[mask]
    return ds.sel(time=time_subset)


## Set the domain of constraints
def set_constraint(obs:xr.Dataset, domain:list, buffer:int=10, reso:int=1):
    # Currently only support data that do not cross the 0 degree longitude
    lonw, lone, lats, latn = domain
    return obs.sel(lon=slice(lonw-buffer*reso, lone+buffer*reso), 
                   lat=slice(lats-buffer*reso, latn+buffer*reso))


# Replace Obs with climatology for the input
def set_constraint_weights(constraint_weights_template:xr.DataArray, buffer:int=10):
    # buffer: The number of grid points that the buffer zone spans
    assert constraint_weights_template.ndim == 2
    n_lat, n_lon = constraint_weights_template.shape
    
    assert n_lat // 2 > buffer and n_lon // 2 > buffer
    assert 100 % buffer == 0
    delta_weight = 1 / buffer  # 0.2

    constraint_weights = constraint_weights_template.copy()
    constraint_weights[:] = np.zeros((n_lat, n_lon))

    for i_buffer in range(buffer+1):
        constraint_weight = i_buffer * delta_weight  
        constraint_weights[i_buffer:-i_buffer, i_buffer:-i_buffer] = constraint_weight  
        
    return constraint_weights
