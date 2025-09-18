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
dir_fig = '/path/to/figures/'

startdates = pd.date_range(start="2021-06-19", end="2021-06-28", freq="D")
startdates = [d.strftime("%Y-%m-%dT00") for d in startdates]
enddate = '2021-06-29T00'

level = 850
target_domain = [230, 250, 40, 60]  # domain to calculate regional average
lonw, lone, lats, latn = target_domain  
g = 9.80665

eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates[0]}_{enddate}.nc')
time_obs = np.datetime64(startdates[0]) + eval_targets['time'].values 

t_free_avg, t_constrained_avg, t_obs_avg = [], [], []
print(startdates)
for startdate in startdates:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    t_free = predictions_free['temperature'][:, 0].loc[:, level] - 273.15
    t_constrained = predictions_constrained['temperature'][:, 0].loc[:, level] - 273.15
    time_forecast = np.datetime64(startdate) + t_free['time'].values
    t_free['time'] = time_forecast
    t_constrained['time'] = time_forecast
    time_loc = np.isin(time_obs, time_forecast)
    print(startdate, time_loc.sum())
    t_obs = eval_targets['temperature'][0, time_loc].loc[:, level] - 273.15
    t_obs['time'] = time_forecast

    t_free_avg.append(af.weighted_avg(t_free.loc[:, lats:latn, lonw:lone]))
    t_constrained_avg.append(af.weighted_avg(t_constrained.loc[:, lats:latn, lonw:lone]))
    t_obs_avg.append(af.weighted_avg(t_obs.loc[:, lats:latn, lonw:lone]).values)
    
t_free_error_avg = [fct - obs for fct, obs in zip(t_free_avg, t_obs_avg)]
t_constrained_error_avg = [fct - obs for fct, obs in zip(t_constrained_avg, t_obs_avg)]

lonlim, latlim = [90, 260], [0, 80]

startdate = '2021-06-21T00'
predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
step_diff = eval_targets['time'].size - predictions_free['time'].size
constrains_domain = predictions_constrained.attrs['constrains_domain']

step = 32  # 8 days
final_step = np.timedelta64(step*6, 'h')
final_date = np.datetime64(f'{startdate}:00:00') + final_step
final_date = final_date.astype(datetime.datetime).strftime('%Y-%m-%dT%H')

t_free = predictions_free['temperature'][step-1, 0].loc[level] - 273.15  
t_constrained = predictions_constrained['temperature'][step-1, 0].loc[level] - 273.15  
t_obs = eval_targets['temperature'][0, step+step_diff-1].loc[level] - 273.15  
t_free = af.sellonlat(t_free, lonlim+latlim)
t_constrained = af.sellonlat(t_constrained, lonlim+latlim)
t_obs = af.sellonlat(t_obs, lonlim+latlim)

t_constrained_error = t_constrained - t_obs
t_free_error = t_free - t_obs
error_reduction = np.abs(t_constrained_error) - np.abs(t_free_error) 

levelz = 500  # for z
z_free = predictions_free['geopotential'][step-1, 0].loc[levelz] / g
z_constrained = predictions_constrained['geopotential'][step-1, 0].loc[levelz] / g 
z_obs = eval_targets['geopotential'][0, step+step_diff-1].loc[levelz] / g 
z_free = af.sellonlat(z_free, lonlim+latlim)
z_constrained = af.sellonlat(z_constrained, lonlim+latlim)
z_obs = af.sellonlat(z_obs, lonlim+latlim)
z_error_free = z_free - z_obs
z_error_constrained = z_constrained - z_obs

q1_free = xr.open_dataarray(dir_output+f'Q1_free_{startdate}_{enddate}.nc')
q1_constrained = xr.open_dataarray(dir_output+f'Q1_constrained_{startdate}_{enddate}.nc')

z500_anom_free = xr.open_dataarray(dir_output+f'z500_anom_free_{startdate}_{enddate}.nc') / g
z500_anom_constrained = xr.open_dataarray(dir_output+f'z500_anom_constrained_{startdate}_{enddate}.nc') / g

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

vmax = 10
levels1 = np.linspace(-8, 8, 17)
levels2 = np.linspace(-vmax, vmax, 21) 
cmap1 = eplt.cmap_white_center(levels1, cmap='PRGn_r')  
cmap2 = eplt.cmap_white_center(levels2) 
unit = 'K'
varname = 'T850'
target_color = 'gold'

central_longitude = 180
proj = ccrs.PlateCarree(central_longitude=central_longitude)

fig, axs = eplt.subplots(ncols=2, nrows=3, figsize=[12., 12], 
                         proj=(None, proj, proj, proj, proj, proj), wspace=0.15, hspace=0.2)  #  
eplt.formats(axs, geo=True, abc='(a)', lonlim=lonlim, latlim=latlim, # reso='med', 
             lonloc=30, latloc=20)

ax1 = axs[0, 0]
eplt.formats(ax1, xtick_params={'rotation':45}, xlabel='Valid time', ylabel=f'{varname} ({unit})', ylim=[-0.5, 9.5], 
             title='Forecast Absolute Error Growth')
for i, t_error in enumerate(t_free_error_avg):
    # print(t_error.values)
    ax1.plot(t_error.time, np.abs(t_error), ms=3, label=startdates[i])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.legend(loc='upper left', fontsize=9, ncol=2, title='Initialization time')

# Reset the color cycle
ax1.set_prop_cycle(None)
for i, t_error in enumerate(t_constrained_error_avg):
    ax1.plot(t_error.time, np.abs(t_error), '--', ms=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
pos = ax1.get_position()
adjust_height = 0.057
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])

lw = 0.7
colorbar_kw = dict(aspect=35, pad=0.1, shrink=0.9, ticklength=3)

ax1 = axs[0,1]
eplt.formats(ax1, title=f'Error Reduction  (Lead = 8 days)')
pic = eplt.contourf(ax1, error_reduction, levels=levels1, cmap=cmap1, globe=True)  
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::2], label=f'{varname} absolute error difference ({unit})', **colorbar_kw) 

## Error 
levelz_pos = np.arange(40, 330, 40)
levelz_neg = np.arange(-320, 0, 40)
ax1 = axs[1,0]
eplt.formats(ax1, title=f'Free Forecast Error  (Lead = 8 days)')
pic = eplt.contourf(ax1, t_free_error, cmap=cmap2, levels=levels2, globe=True)
eplt.contour(ax1, z_error_free, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k')
eplt.contour(ax1, z_error_free, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k')
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::2], label=f'{varname} error ({unit})', **colorbar_kw) 

ax2 = axs[1,1]
eplt.formats(ax2, title=f'TSC Forecast Error  (Lead = 8 days)')
pic = eplt.contourf(ax2, t_constrained_error, cmap=cmap2, levels=levels2, globe=True)
eplt.contour(ax2, z_error_constrained, level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k')
eplt.contour(ax2, z_error_constrained, level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k')
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::2], label=f'{varname} error ({unit})', **colorbar_kw) 

## Q1 & Z500_anomalies
step = 16
time_q1 = q1_free['time'][step-1] + np.datetime64(startdate)
time_q1.dt.strftime('%Y-%m-%dT%H')

levels_q1 = np.linspace(100, 1600, 16)
cmap_q1 = plt.get_cmap('GnBu')
cmap_q1.set_under('w')

ax1 = axs[2,0]
eplt.formats(ax1, title='Free Forecast  (Lead = 4 days)')
pic = eplt.contourf(ax1, q1_free[step-1], cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax1, z500_anom_free[step-1], level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax1, z500_anom_free[step-1], level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

ax2 = axs[2,1]
eplt.formats(ax2, title='TSC Forecast  (Lead = 4 days)')
pic = eplt.contourf(ax2, q1_constrained[step-1], cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax2, z500_anom_constrained[step-1], level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax2, z500_anom_constrained[step-1], level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

fig.savefig(dir_fig+'figure5.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure5.pdf', dpi=300, bbox_inches='tight')
