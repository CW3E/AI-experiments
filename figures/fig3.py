import os
os.chdir('/expanse/nfs/cw3e/cwp179/')
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

dir_era5 = '/path/to/ERA5/'
dir_output = '/path/to/graphcast/output/'
dir_fig = '/path/to/figures/'

startdates = pd.date_range(start="2016-03-03", end="2016-03-12", freq="D")
startdates = [d.strftime("%Y-%m-%dT00") for d in startdates]
enddate = '2016-03-13T00'

level = 500
target_domain = [0, 40, 55, 77]  # domain to calculate regional average
lonw, lone, lats, latn = target_domain  
g = 9.80665

eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates[0]}_{enddate}.nc')
time_obs = np.datetime64(startdates[0]) + eval_targets['geopotential']['time'].values

z_free_avg, z_constrained_avg, z_obs_avg = [], [], []
print(startdates)
for startdate in startdates:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    z_free = predictions_free['geopotential'][:, 0].loc[:, level] / g
    z_constrained = predictions_constrained['geopotential'][:, 0].loc[:, level] / g
    time_forecast = np.datetime64(startdate) + z_free['time'].values
    z_free['time'] = time_forecast
    z_constrained['time'] = time_forecast
    time_loc = np.isin(time_obs, time_forecast)
    print(startdate, time_loc.sum())
    z_obs = eval_targets['geopotential'][0, time_loc].loc[:, level] / g
    z_obs['time'] = time_forecast

    z_free_avg.append(af.weighted_avg(z_free.loc[:, lats:latn, lonw:lone]))
    z_constrained_avg.append(af.weighted_avg(z_constrained.loc[:, lats:latn, lonw:lone]))
    z_obs_avg.append(af.weighted_avg(z_obs.loc[:, lats:latn, lonw:lone]).values)
    
z_free_error_avg = [fct - obs for fct, obs in zip(z_free_avg, z_obs_avg)]
z_constrained_error_avg = [fct - obs for fct, obs in zip(z_constrained_avg, z_obs_avg)]

lonlim, latlim = [-100, 55], [10, 90]

startdate = '2016-03-05T00'
predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
step_diff = eval_targets['time'].size - predictions_free['time'].size
constrains_domain = predictions_constrained.attrs['constrains_domain']

step = 32  # 8 days
final_step = np.timedelta64(step*6, 'h')
final_date = np.datetime64(f'{startdate}:00:00') + final_step
final_date = final_date.astype(datetime.datetime).strftime('%Y-%m-%dT%H')

z_free = predictions_free['geopotential'][step-1, 0].loc[level] / g
z_constrained = predictions_constrained['geopotential'][step-1, 0].loc[level] / g
z_obs = eval_targets['geopotential'][0, step+step_diff-1].loc[level] / g  
z_free = af.sellonlat(z_free, lonlim+latlim)
z_constrained = af.sellonlat(z_constrained, lonlim+latlim)
z_obs = af.sellonlat(z_obs, lonlim+latlim)

z_constrained_error = z_constrained - z_obs
z_free_error = z_free - z_obs
error_reduction = np.abs(z_constrained_error) - np.abs(z_free_error) 

q1_free = xr.open_dataarray(dir_output+f'Q1_free_{startdate}_{enddate}.nc')
q1_constrained = xr.open_dataarray(dir_output+f'Q1_constrained_{startdate}_{enddate}.nc')

z500_anom_free = xr.open_dataarray(dir_output+f'z500_anom_free_{startdate}_{enddate}.nc') / g
z500_anom_constrained = xr.open_dataarray(dir_output+f'z500_anom_constrained_{startdate}_{enddate}.nc') / g

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

vmax = 300
levels = np.linspace(-vmax, vmax, 21)
cmap1 = eplt.cmap_white_center(levels, cmap='PRGn_r')
cmap2 = eplt.cmap_white_center(levels, cmap='RdBu_r')
unit = 'gpm'
varname = 'Z500'
target_color = 'gold'

central_longitude = 0
proj = ccrs.PlateCarree(central_longitude=central_longitude)

fig, axs = eplt.subplots(ncols=2, nrows=3, figsize=[11.5, 12], 
                         proj=(None, proj, proj, proj, proj, proj), wspace=0.15, hspace=0.15)  #  
eplt.formats(axs, geo=True, abc='(a)', lonlim=lonlim, latlim=latlim, # reso='med', 
             lonloc=30, latloc=20)

ax1 = axs[0, 0]
eplt.formats(ax1, xtick_params={'rotation':45}, xlabel='Valid time', ylabel=f'{varname} ({unit})', ylim=[-10, 300], 
             title='Forecast Absolute Error Growth')
for i, z_error in enumerate(z_free_error_avg):
    ax1.plot(z_error.time, np.abs(z_error), ms=3, label=startdates[i])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.legend(loc='upper left', fontsize=9, ncol=2, title='Initialization time')

# Reset the color cycle
ax1.set_prop_cycle(None)
for i, z_error in enumerate(z_constrained_error_avg):
    ax1.plot(z_error.time, np.abs(z_error), '--', ms=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

pos = ax1.get_position()
adjust_height = 0.059
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])

colorbar_kw = dict(aspect=35, pad=0.1, shrink=0.9, ticklength=3)


ax3 = axs[0,1]
eplt.formats(ax3, title=f'Error Reduction  (Lead = 8 days)')
pic = eplt.contourf(ax3, error_reduction, levels=levels, cmap=cmap1, globe=True)  
eplt.addpatch(ax3, target_domain, lw=1.5, ec=target_color)  # 'tab:purple'
eplt.addpatch(ax3, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax3, pic, ticks=pic.levels[::4], label=f'{varname} absolute error difference ({unit})', **colorbar_kw) 

ax1 = axs[1,0]
eplt.formats(ax1, title=f'Free Forecast Error  (Lead = 8 days)')
pic = eplt.contourf(ax1, z_free_error, cmap=cmap2, levels=levels, globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::4], label=f'{varname} error ({unit})', **colorbar_kw) 

ax2 = axs[1,1]
eplt.formats(ax2, title=f'TSC Forecast Error  (Lead = 8 days)')
pic = eplt.contourf(ax2, z_constrained_error, cmap=cmap2, levels=levels, globe=True)
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::4], label=f'{varname} error ({unit})', **colorbar_kw) 


step = 20

levels_q1 = np.linspace(100, 1600, 16)
cmap_q1 = plt.get_cmap('GnBu')
cmap_q1.set_under('w')

levelz_pos = np.arange(100, 550, 50)
levelz_neg = np.arange(-500, -50, 50)
lw = 0.7

ax1 = axs[2,0]
eplt.formats(ax1, title='Free Forecast  (Lead = 5 days)')
pic = eplt.contourf(ax1, q1_free[step-1], cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax1, z500_anom_free[step-1], level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax1, z500_anom_free[step-1], level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

ax2 = axs[2,1]
eplt.formats(ax2, title='TSC Forecast  (Lead = 5 days)')
pic = eplt.contourf(ax2, q1_constrained[step-1], cmap=cmap_q1, levels=levels_q1, globe=True)
eplt.contour(ax2, z500_anom_constrained[step-1], level=levelz_pos, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.contour(ax2, z500_anom_constrained[step-1], level=levelz_neg, clabel=False, fmt='%i', lw=lw, color='k', globe=True)
eplt.addpatch(ax2, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax2, constrains_domain, lw=1.5, ec='r')
eplt.addcolorbar(ax2, pic, ticks=pic.levels[::3], label='Q$_{1}$ (W m$^{-2}$)', **colorbar_kw) 

fig.savefig(dir_fig+'figure3.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure3.pdf', dpi=300, bbox_inches='tight')
