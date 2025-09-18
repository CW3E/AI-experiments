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

startdates = pd.date_range(start="2016-03-03T00", end="2016-03-12T18", freq="6H")
startdates = [d.strftime("%Y-%m-%dT%H") for d in startdates] 
enddate = '2016-03-13T00'

# Forecast skills as a function of initialization time
i_step = -1
level = 500
lonlim, latlim = [-25, 65], [35, 90]
target_domain = [0, 40, 55, 77]  # domain to calculate regional average
lonw, lone, lats, latn = target_domain  
g = 9.80665

eval_targets = xr.open_dataset(dir_output+f'ERA5_{startdates[0]}_{enddate}.nc')
z_obs = eval_targets['geopotential'][0, i_step].loc[level] / g  

# climatology 
z_clim = xr.open_dataset(dir_era5+f'z500_{enddate[5:]}_clim.nc')['z500'][0] / g
z_clim = z_clim.reindex(latitude=z_clim.latitude[::-1])
z_clim = z_clim.loc[z_obs['lat'].values, z_obs['lon'].values]

z_obs_anom = z_obs - z_clim.values
z_obs_anom = af.sellonlat(z_obs_anom, lonlim+latlim)
z_obs = af.sellonlat(z_obs, lonlim+latlim)
z_obs_avg = af.weighted_avg(z_obs.loc[lats:latn, lonw:lone])

# forecast skills 
z_free_avg, z_constrained_avg = [], []
for startdate in startdates:        
    predictions_free = xr.open_dataset(dir_output+f'predictions_free_{startdate}_{enddate}.nc')
    predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
    z_free = predictions_free['geopotential'][i_step, 0].loc[level] / g
    z_constrained = predictions_constrained['geopotential'][i_step, 0].loc[level] / g
    z_free_avg.append(af.weighted_avg(z_free.loc[lats:latn, lonw:lone]).values)
    z_constrained_avg.append(af.weighted_avg(z_constrained.loc[lats:latn, lonw:lone]).values)

xticklabels = pd.to_datetime(startdates).strftime('%m-%d').tolist()
xticklabels = xticklabels[::4]  # select 00 UTC

# IFS
z_ifs_avg = []
for startdate in startdates[::2]:  # only 00 and 12 UTC
    # print(startdate)
    predictions_ifs = xr.open_dataset(dir_ifs+f'{startdate.replace('T', '_')}/z500_ctl_{startdate}.grib', 
                                    engine='cfgrib', backend_kwargs={"indexpath": ""})
    valid_time = predictions_ifs['valid_time'].dt.strftime("%Y-%m-%dT%H") 
    z_ifs = predictions_ifs['gh'][valid_time == enddate].squeeze()
    z_ifs_avg.append(af.weighted_avg(z_ifs.loc[latn:lats, lonw:lone]).values)


## Plot
vmax = 300
levels = np.linspace(-vmax, vmax, 21)
cmap = eplt.cmap_white_center(levels, cmap='RdBu_r')

central_longitude = 0
proj = ccrs.PlateCarree(central_longitude=central_longitude)
colorbar_kw = dict(aspect=25, pad=0.1, shrink=0.9, ticklength=3)


fig, axs = eplt.subplots(ncols=2, nrows=1, figsize=[12, 4.5], 
                         proj=(proj, None), hspace=0.15)
# fig, axs = eplt.subplots(ncols=2, nrows=2, figsize=[12, 9.5], 
                        #  proj=(proj, None, proj, proj), hspace=0.15)
eplt.formats(axs, geo=True, abc='(a)', lonlim=lonlim, latlim=latlim, reso='med')

# observation
ax1 = axs[0] # axs[0,0]
eplt.formats(ax1, title='ERA5')
pic = eplt.contourf(ax1, z_obs_anom, cmap=cmap, levels=levels, globe=True)
eplt.contour(ax1, z_obs, level=np.arange(4800, 5900, 100), clabel=True, fmt='%i', lw=1., globe=True)
eplt.addcolorbar(ax1, pic, ticks=pic.levels[::4], label=f'Z500 anomaly (gpm)', **colorbar_kw)  # 
eplt.addpatch(ax1, target_domain, lw=1.5, ec='gold') # tab:purple

z500_p95 = xr.open_dataset(dir_era5+f'z.500.anom.thresholds.nc')['p95']
z500_p95 = af.sellonlat(z500_p95, lonlim+latlim)
z500_p95 = z500_p95.rename({'latitude': 'lat', 'longitude': 'lon'})
z_mask = xr.where(z_obs_anom >= z500_p95, 0, 1)
plt.rcParams['hatch.linewidth'] = 0.5
eplt.plt_sig(ax1, z_mask, color='0.7', alpha=0.8, method=1, size=2, hatches='///',)


xticklabels = pd.to_datetime(startdates).strftime('%m-%d').tolist()
xticklabels = xticklabels[::4]  # select 00 UTC

ax1 = axs[1] # axs[0,1]
eplt.formats(ax1, title='Forecasted Z500', xticks=np.arange(0, len(startdates), 4),  #np.arange(0, len(startdates), 1)
             xticklabels=xticklabels, xtick_params={'rotation':45}, 
             xlabel='Initialization time', ylabel='gpm', ylim=[5250, 5580])
ax1.plot(np.arange(len(startdates)), z_free_avg, 'o-', label='GraphCast Free', ms=3)
ax1.plot(np.arange(len(startdates)), z_constrained_avg, 'o-', label='GraphCast TSC', ms=3)
ax1.plot(np.arange(0, len(startdates), 2), z_ifs_avg, 'o-', label='IFS', ms=3, zorder=0)
# ax1.plot(np.arange(len(startdates[::4])), z_free_avg[::4], 'o-', label='Free', ms=4)
# ax1.plot(np.arange(len(startdates[::4])), z_constrained_avg[::4], 'o-', label='TSC', ms=4)
ax1.axhline(z_obs_avg, color='k', lw=1.5)
ax1.legend(loc='lower right')  # , fontsize=10
# ax1.set_box_aspect(0.6)
pos = ax1.get_position()
adjust_height = 0.192  # 0.09
ax1.set_position([pos.x0, pos.y0+adjust_height, pos.width, pos.y1-pos.y0-adjust_height])

fig.savefig(dir_fig+'figure2.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure2.pdf', dpi=300, bbox_inches='tight')
