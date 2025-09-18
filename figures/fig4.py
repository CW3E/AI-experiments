import sys
sys.path.append('../utils/')
import numpy as np
import xarray as xr
import pandas as pd
import array_func as af
import earthplot as eplt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dir_era5 = '/path/to/ERA5/'
dir_output = '/path/to/graphcast/output/'
dir_ifs = '/path/to/IFS/'
dir_fig = '/path/to/figures/'

startdates = pd.date_range(start="2021-06-19T00", end="2021-06-28T18", freq="6H")
startdates = [d.strftime("%Y-%m-%dT%H") for d in startdates]
enddate = '2021-06-29T00'

# Forecast skills as a function of initialization time
i_step = -1
level = 500  # for z    
lonlim, latlim = [190, 280], [25, 80]
target_domain = [230, 250, 40, 60]  # domain to calculate regional average
lonw, lone, lats, latn = target_domain  
# plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab10').colors)

g = 9.80665
eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates[0]}_{enddate}.nc')
z_obs = eval_targets['geopotential'][0, i_step].loc[level] / g  
z_obs = af.sellonlat(z_obs, lonlim+latlim)
t_obs = eval_targets['2m_temperature'][0, i_step] - 273.15

# climatology 
t_clim = xr.open_dataset(dir_era5+f'/tas_{enddate[5:]}_clim.nc')['tas'][0] - 273.15
t_clim = t_clim.reindex(latitude=t_clim.latitude[::-1])
t_clim = t_clim.loc[t_obs['lat'].values, t_obs['lon'].values]

t_obs_anom = t_obs - t_clim.values
t_obs_anom = af.sellonlat(t_obs_anom, lonlim+latlim)
t_obs = af.sellonlat(t_obs, lonlim+latlim)
t_obs_avg = af.weighted_avg(t_obs.loc[lats:latn, lonw:lone])

# GraphCast forecast skills 
t_free_avg, t_constrained_avg = [], []
for startdate in startdates:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    t_free = predictions_free['2m_temperature'][i_step, 0] - 273.15
    t_constrained = predictions_constrained['2m_temperature'][i_step, 0] - 273.15
    t_free_avg.append(af.weighted_avg(t_free.loc[lats:latn, lonw:lone]).values)
    t_constrained_avg.append(af.weighted_avg(t_constrained.loc[lats:latn, lonw:lone]).values)

# IFS
t_ifs_avg = []
for startdate in startdates[::2]:  # only 00 and 12 UTC
    # print(startdate)
    predictions_ifs = xr.open_dataset(dir_ifs+f'{startdate.replace('T', '_')}/t2m_ctl_{startdate}.grib', 
                                    engine='cfgrib', backend_kwargs={"indexpath": ""})
    valid_time = predictions_ifs['valid_time'].dt.strftime("%Y-%m-%dT%H") 
    t_ifs = predictions_ifs['t2m'][valid_time == enddate].squeeze()
    t_ifs_avg.append(af.weighted_avg(t_ifs.loc[latn:lats, lonw:lone]).values - 273.15)


## plot
vmax = 15
unit = '℃'
levels = np.linspace(-vmax, vmax, 16) #np.linspace(-vmax, vmax, 21)
z_levels = np.arange(4800, 6000, 50)
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab10').colors)

## plot
central_longitude = 180
proj = ccrs.PlateCarree(central_longitude=central_longitude)
colorbar_kw = dict(aspect=30, pad=0.1, shrink=0.9, ticklength=3)

fig, axs = eplt.subplots(ncols=2, nrows=1, figsize=[12, 4.5], 
                         proj=(proj, None), hspace=0.15)
# fig, axs = eplt.subplots(ncols=2, nrows=2, figsize=[12, 9.5], 
                        #  proj=(proj, None, proj, proj), hspace=0.15)
eplt.formats(axs, geo=True, abc='(a)', lonlim=lonlim, latlim=latlim, reso='med')

# observation
ax1 = axs[0] # axs[0,0]
eplt.formats(ax1, title='ERA5')
pic = eplt.contourf(ax1, t_obs_anom, cmap='RdBu_r', levels=levels, globe=True)
eplt.contour(ax1, z_obs, level=z_levels, clabel=True, fmt='%i', lw=1., globe=True)
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::3], label=f'T2m anomaly ({unit})', **colorbar_kw)  # 
eplt.addpatch(ax1, target_domain, lw=1.5, ec='gold') #'tab:purple'

t2m_p95 = xr.open_dataset(dir_era5+f'/t2m.anom.thresholds.nc')['p99']
t2m_p95 = af.sellonlat(t2m_p95, lonlim+latlim)
t2m_p95 = t2m_p95.rename({'latitude': 'lat', 'longitude': 'lon'})
t_mask = xr.where(t_obs_anom >= t2m_p95.values, 0, 1)
plt.rcParams['hatch.linewidth'] = 0.5
eplt.plt_sig(ax1, t_mask, color='0.7', alpha=0.8, method=1, size=2, hatches='///',)


xticklabels = pd.to_datetime(startdates).strftime('%m-%d').tolist()
xticklabels = xticklabels[::4]  # select 00 UTC

ax1 = axs[1] # axs[0,1]
eplt.formats(ax1, title='Forecasted T2m', xticks=np.arange(0, len(startdates), 4), # np.arange(0, len(startdates), 1)
             xticklabels=xticklabels, xtick_params={'rotation':45}, 
             xlabel='Initialization time', ylabel=unit)  # , ylim=[5250, 5580]
ax1.plot(np.arange(len(startdates)), t_free_avg, 'o-', label='GraphCast Free', ms=3)
ax1.plot(np.arange(len(startdates)), t_constrained_avg, 'o-', label='GraphCast TSC', ms=3)
ax1.plot(np.arange(0, len(startdates), 2), t_ifs_avg, 'o-', label='IFS', ms=3, zorder=0)
ax1.axhline(t_obs_avg, color='k', lw=1.5)
ax1.legend(loc='lower right')  # , fontsize=10
# ax1.set_box_aspect(0.6)
pos = ax1.get_position()
adjust_height = 0.191 # 0.09 
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])


fig.savefig(dir_fig+'figure4.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure4.pdf', dpi=300, bbox_inches='tight')
