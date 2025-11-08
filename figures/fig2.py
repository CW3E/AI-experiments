import sys
sys.path.append('../utils/')
import numpy as np
import xarray as xr
import pandas as pd
import array_func as af
import earthplot as eplt
import cartopy.crs as ccrs
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from cycler import cycler
import warnings
warnings.filterwarnings("ignore")

dir_era5 = '/path/to/ERA5/'
dir_output = '/path/to/graphcast/output/'
dir_ifs = '/path/to/IFS/'
dir_fig = '/path/to/figures/'

lev_z = 500  # for z    
lonlim1, latlim1 = [190, 280], [30, 75]  # extent of panel a  # 90:45
lonlim2, latlim2 = [95, 255], [0, 80]   # extent of panels d-i
target_domain = [230, 250, 40, 60]  # domain to calculate regional average
lonw, lone, lats, latn = target_domain  
g = 9.80665

# For panel a: Forecast skills as a function of initialization time
startdates_6h = pd.date_range(start="2021-06-19T00", end="2021-06-28T18", freq="6H")
startdates_6h = [d.strftime("%Y-%m-%dT%H") for d in startdates_6h]
enddate = '2021-06-29T00'
i_step = -1

# ERA5 observations of the event onset date
eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates_6h[0]}_{enddate}.nc')
z_obs_onset = eval_targets['geopotential'][0, i_step].loc[lev_z] / g  
z_obs_onset = af.sellonlat(z_obs_onset, lonlim1+latlim1)
t_obs_onset = eval_targets['2m_temperature'][0, i_step] - 273.15

t_clim = xr.open_dataset(dir_era5+f'tas_{enddate[5:]}_clim.nc')['tas'][0] - 273.15
t_clim = t_clim.reindex(latitude=t_clim.latitude[::-1])
t_clim = t_clim.loc[t_obs_onset['lat'].values, t_obs_onset['lon'].values]

t_obs_onset_anom = t_obs_onset - t_clim.values
t_obs_onset_anom = af.sellonlat(t_obs_onset_anom, lonlim1+latlim1)
t_obs_onset = af.sellonlat(t_obs_onset, lonlim1+latlim1)
t_obs_onset_avg = af.weighted_avg(t_obs_onset.loc[lats:latn, lonw:lone])

# GraphCast forecast skills 
t_free_onset_avg, t_constrained_onset_avg = [], []
for startdate in startdates_6h:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    t_free = predictions_free['2m_temperature'][i_step, 0] - 273.15
    t_constrained = predictions_constrained['2m_temperature'][i_step, 0] - 273.15
    t_free_onset_avg.append(af.weighted_avg(t_free.loc[lats:latn, lonw:lone]).values)
    t_constrained_onset_avg.append(af.weighted_avg(t_constrained.loc[lats:latn, lonw:lone]).values)

# IFS forecast skills 
t_ifs_onset_avg = []
for startdate in startdates_6h[::2]:  # only 00 and 12 UTC
    # print(startdate)
    predictions_ifs = xr.open_dataset(dir_ifs+f'{startdate.replace('T', '_')}/t2m_ctl_{startdate}.grib', 
                                    engine='cfgrib', backend_kwargs={"indexpath": ""})
    valid_time = predictions_ifs['valid_time'].dt.strftime("%Y-%m-%dT%H") 
    t_ifs = predictions_ifs['t2m'][valid_time == enddate].squeeze()
    t_ifs_onset_avg.append(af.weighted_avg(t_ifs.loc[latn:lats, lonw:lone]).values - 273.15)

# For errors
startdates = pd.date_range(start="2021-06-19", end="2021-06-28", freq="D")
startdates = [d.strftime("%Y-%m-%dT00") for d in startdates]
enddate = '2021-06-29T00'
lev_t = 850

# Error growth curves
eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates[0]}_{enddate}.nc')
time_obs = np.datetime64(startdates[0]) + eval_targets['time'].values 

t_free_avg, t_constrained_avg, t_obs_avg = [], [], []
print(startdates)
for startdate in startdates:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    t_free = predictions_free['temperature'][:, 0].loc[:, lev_t] #- 273.15
    t_constrained = predictions_constrained['temperature'][:, 0].loc[:, lev_t] #- 273.15
    time_forecast = np.datetime64(startdate) + t_free['time'].values
    t_free['time'] = time_forecast
    t_constrained['time'] = time_forecast
    time_loc = np.isin(time_obs, time_forecast)
    print(startdate, time_loc.sum())
    t_obs = eval_targets['temperature'][0, time_loc].loc[:, lev_t] #- 273.15
    t_obs['time'] = time_forecast

    t_free_avg.append(af.weighted_avg(t_free.loc[:, lats:latn, lonw:lone]))
    t_constrained_avg.append(af.weighted_avg(t_constrained.loc[:, lats:latn, lonw:lone]))
    t_obs_avg.append(af.weighted_avg(t_obs.loc[:, lats:latn, lonw:lone]).values)
    
t_free_error_avg = [fct - obs for fct, obs in zip(t_free_avg, t_obs_avg)]
t_constrained_error_avg = [fct - obs for fct, obs in zip(t_constrained_avg, t_obs_avg)]

# Error spatial distributions

startdate = '2021-06-21T00'
predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
step_diff = eval_targets['time'].size - predictions_free['time'].size
constrains_domain = predictions_constrained.attrs['constrains_domain']

step = 32  # 8 days
final_step = np.timedelta64(step*6, 'h')
final_date = np.datetime64(f'{startdate}:00:00') + final_step
final_date = final_date.astype(datetime.datetime).strftime('%Y-%m-%dT%H')

t_free = predictions_free['temperature'][step-1, 0].loc[lev_t] #- 273.15  # 
t_constrained = predictions_constrained['temperature'][step-1, 0].loc[lev_t] #- 273.15  # 
t_obs = eval_targets['temperature'][0, step+step_diff-1].loc[lev_t] #- 273.15  # 
t_free = af.sellonlat(t_free, lonlim2+latlim2)
t_constrained = af.sellonlat(t_constrained, lonlim2+latlim2)
t_obs = af.sellonlat(t_obs, lonlim2+latlim2)

t_constrained_error = t_constrained - t_obs
t_free_error = t_free - t_obs
error_reduction = np.abs(t_constrained_error) - np.abs(t_free_error) 
# error_reduction_rate = (np.abs(t_free - t_obs) - np.abs(t_constrained - t_obs)) / np.abs(t_free - t_obs) * 100

z_free = predictions_free['geopotential'][step-1, 0].loc[lev_z] / g
z_constrained = predictions_constrained['geopotential'][step-1, 0].loc[lev_z] / g 
z_obs = eval_targets['geopotential'][0, step+step_diff-1].loc[lev_z] / g 
z_free = af.sellonlat(z_free, lonlim2+latlim2)
z_constrained = af.sellonlat(z_constrained, lonlim2+latlim2)
z_obs = af.sellonlat(z_obs, lonlim2+latlim2)
z_error_free = z_free - z_obs
z_error_constrained = z_constrained - z_obs

# Dynamics: Q1

step = 16

q1_free = xr.open_dataarray(dir_output+f'Q1_free_{startdate}_{enddate}.nc')[step-1]
q1_constrained = xr.open_dataarray(dir_output+f'Q1_constrained_{startdate}_{enddate}.nc')[step-1]

z500_anom_free = xr.open_dataarray(dir_output+f'z500_anom_free_{startdate}_{enddate}.nc')[step-1] / g
z500_anom_constrained = xr.open_dataarray(dir_output+f'z500_anom_constrained_{startdate}_{enddate}.nc')[step-1] / g

q1_era5 = xr.open_dataarray(dir_output+f'Q1_ERA5_2021-06-19T00_{enddate}.nc')[step+8-1]
z500_anom_era5 = xr.open_dataarray(dir_output+f'z500_anom_ERA5_2021-06-19T00_{enddate}.nc')[step+8-1] / g


## Plot skills, errors, and dynamics

vmax = 15
unit = '℃'
levels = np.linspace(-vmax, vmax, 16) #np.linspace(-vmax, vmax, 21)
z_levels = np.arange(4800, 6000, 50)
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

## plot
central_longitude = 180
proj = ccrs.PlateCarree(central_longitude=central_longitude)
colorbar_kw = dict(aspect=30, pad=0.1, shrink=0.9, ticklength=3)

fig, axs = eplt.subplots(ncols=3, nrows=3, figsize=[16, 11.], 
                         proj=(proj, None, None, proj, proj, proj, proj, proj, proj), wspace=0.15) # hspace=0.15,
# fig, axs = eplt.subplots(ncols=2, nrows=2, figsize=[12, 9.5], 
                        #  proj=(proj, None, proj, proj), hspace=0.15)
eplt.formats(axs, abc='(a)',)
eplt.formats(axs[0, 0],  geo=True, lonlim=lonlim1, latlim=latlim1) # reso='med'
eplt.formats(axs[1:, :], geo=True, lonlim=lonlim2, latlim=latlim2, lonloc=30, latloc=20) # reso='med'

# (a) Observed T2m anomaly
ax1 = axs[0,0]
eplt.formats(ax1, title='ERA5  (2021-06-29T00)')
pic = eplt.contourf(ax1, t_obs_onset_anom, cmap='RdBu_r', levels=levels, globe=True)
eplt.contour(ax1, z_obs_onset, level=z_levels, clabel=True, fmt='%i', lw=1., globe=True)
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label=f'T2m anomaly ({unit})', **colorbar_kw)  # 
eplt.addpatch(ax1, target_domain, lw=1.5, ec='gold') #'tab:purple'

t2m_p95 = xr.open_dataset(dir_root+f'data/ERA5/anomaly/t2m.anom.thresholds.nc')['p99']
t2m_p95 = af.sellonlat(t2m_p95, lonlim1+latlim1)
t2m_p95 = t2m_p95.rename({'latitude': 'lat', 'longitude': 'lon'})
t_mask = xr.where(t_obs_onset_anom >= t2m_p95.values, 0, 1)
plt.rcParams['hatch.linewidth'] = 0.5
eplt.plt_sig(ax1, t_mask, color='0.7', alpha=0.8, method=1, size=2, hatches='///',)

# (b) Forecast skills of T2m
xticklabels = pd.to_datetime(startdates_6h).strftime('%m-%d').tolist()
xticklabels = xticklabels[::4]  # select 00 UTC

ax1 = axs[0,1] 
markersize = 2.5
eplt.formats(ax1, title='Forecasted T2m', xticks=np.arange(0, len(startdates_6h), 4), # np.arange(0, len(startdates), 1)
             xticklabels=xticklabels, xtick_params={'rotation':45}, 
             xlabel='Initialization time', ylabel=f"T2m ({unit})")  # , ylim=[5250, 5580]
ax1.plot(np.arange(len(startdates_6h)), t_free_onset_avg, 'o-', c='tab:blue', label='GraphCast Free', ms=markersize)
ax1.plot(np.arange(len(startdates_6h)), t_constrained_onset_avg, 'o-', c='tab:red', label='GraphCast TSC', ms=markersize)
ax1.plot(np.arange(0, len(startdates_6h), 2), t_ifs_onset_avg, 'o-', c='tab:green', label='IFS', ms=markersize, zorder=0)
ax1.axhline(t_obs_onset_avg, color='k', lw=1.5)
ax1.legend(loc='lower right')  # , fontsize=10
# ax1.set_box_aspect(0.6)
pos = ax1.get_position()
adjust_height = 0.058 # 0.09 
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])

# (c) Error growth curves
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

vmax = 15
levels1 = np.linspace(-8, 8, 17)
levels2 = np.linspace(-vmax, vmax, 21) #np.hstack((np.arange(-10, 0, 1), np.arange(1, 11, 1))) #np.array([-10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10]) # np.linspace(-vmax, vmax, 21)
cmap1 = eplt.cmap_white_center(levels1, cmap='PRGn_r')  # 'RdBu_r' # cm.get_cmap('RdBu_r', len(levels1) - 1)
cmap2 = eplt.cmap_white_center(levels2) #'RdBu_r' # cm.get_cmap('RdBu_r', len(levels2) - 1) 
# norm2 = BoundaryNorm(boundaries=levels2, ncolors=cmap2.N)
unit = 'K'
varname = 'T850'
target_color = 'gold'

ax1 = axs[0, 2]
eplt.formats(ax1, xtick_params={'rotation':45}, xlabel='Valid time', ylabel=f'{varname} ({unit})', ylim=[-0.5, 9.5], 
             title='Forecast absolute error growth')
for i, t_error in enumerate(t_free_error_avg):
    # print(t_error.values)
    ax1.plot(t_error.time, np.abs(t_error), label=startdates[i])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.legend(loc='upper left', fontsize=9, ncol=2, title='Initialization time')

# Reset the color cycle
ax1.set_prop_cycle(None)
for i, t_error in enumerate(t_constrained_error_avg):
    # print(t_error.values)
    ax1.plot(t_error.time, np.abs(t_error), '--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# ax1.legend(loc='upper left', fontsize=10, ncol=1)
pos = ax1.get_position()
adjust_height = 0.058
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])

## Error spatial distributions
lw = 0.7
colorbar_kw = dict(aspect=35, pad=0.1, shrink=0.9, ticklength=3)

# (d) Error distributions of free forecasts
levelz_pos = np.arange(40, 330, 40)
levelz_neg = np.arange(-320, 0, 40)

ax1 = axs[1,0]
eplt.formats(ax1, title=f'Free forecast error  (Lead = 8 days)')
pic = eplt.contourf(ax1, t_free_error, cmap=cmap2, levels=levels2, globe=True)
eplt.contour(ax1, z_error_free, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k')
eplt.contour(ax1, z_error_free, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k')
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::2], label=f'{varname} error ({unit})', **colorbar_kw) 

# (e) Error distributions of TSC forecasts
ax2 = axs[1,1]
eplt.formats(ax2, title=f'TSC forecast error  (Lead = 8 days)')
pic = eplt.contourf(ax2, t_constrained_error, cmap=cmap2, levels=levels2, globe=True)
eplt.contour(ax2, z_error_constrained, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k')
eplt.contour(ax2, z_error_constrained, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k')
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::2], label=f'{varname} error ({unit})', **colorbar_kw) 

# (f) Error reduction
ax1 = axs[1,2]
eplt.formats(ax1, title=f'Error reduction  (Lead = 8 days)')
pic = eplt.contourf(ax1, error_reduction, levels=levels1, cmap=cmap1, globe=True)  
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::2], label=f'{varname} absolute error difference ({unit})', **colorbar_kw) 

## Q1 & Z500_anomalies
time_q1 = q1_free['time'] + np.datetime64(startdate)
time_q1.dt.strftime('%Y-%m-%dT%H')

levels_q1 = np.linspace(100, 1600, 16)
cmap_q1 = plt.get_cmap('GnBu')
cmap_q1.set_under('w')

# (g) Q1 and Z500_anom of ERA5
ax1 = axs[2,0]
eplt.formats(ax1, title='ERA5  (2021-06-25T00)')
pic = eplt.contourf(ax1, q1_era5, cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax1, z500_anom_era5, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax1, z500_anom_era5, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

# (h) Q1 and Z500_anom of free forecasts
ax1 = axs[2,1]
eplt.formats(ax1, title='Free forecast  (Lead = 4 days)')
pic = eplt.contourf(ax1, q1_free, cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax1, z500_anom_free, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax1, z500_anom_free, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

# (i) Q1 and Z500_anom of TSC forecasts
ax2 = axs[2,2]
eplt.formats(ax2, title='TSC forecast  (Lead = 4 days)')
pic = eplt.contourf(ax2, q1_constrained, cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax2, z500_anom_constrained, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax2, z500_anom_constrained, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

fig.savefig(dir_fig+'figure2.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure2.pdf', dpi=300, bbox_inches='tight')
