import os
import sys
sys.path.append('../utils/')
import numpy as np
import xarray as xr
import pandas as pd
import array_func as af
import earthplot as eplt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

dir_output = '/path/to/graphcast/output/'
dir_fig = '/path/to/figures/'

def composite_diff(arr1, arr2, equal_var=True, nan_policy='omit'):
    arr_diff = arr2.mean(axis=0) - arr1.mean(axis=0)
    p_arr = xr.full_like(arr_diff, np.nan)
    _, p_arr[...] = stats.ttest_ind(arr1, arr2, axis=0, equal_var=equal_var, nan_policy=nan_policy)
    return arr_diff, p_arr

def get_data(startdates, enddates, lev_t=850, lev_z=500):

    t_init_all = []
    t_clim_init_all = []
    z_init_all = []
    z_clim_init_all = []
    t_constrained_all = []
    t_clim_constrained_all = []
    z_constrained_all = []
    z_clim_constrained_all = []

    for startdate, enddate in zip(startdates, enddates):
        print(startdate, enddate)
        input_constrained = xr.open_dataset(dir_output+f'inputs_{startdate}.nc')  # same as input_free
        input_clim_constrained = xr.open_dataset(dir_output+f'inputs_clim-constrained_{startdate}.nc') 
        predictions_constrained = xr.open_dataset(dir_output+f'predictions_constrained_{startdate}_{enddate}.nc')
        # predictions_clim_constrained = xr.open_dataset(dir_output+f'predictions_clim-constrained_{startdate}_{enddate}.nc')
        predictions_clim_constrained = xr.open_dataset(dir_output+f'predictions_clim-constrained_{startdate}_{enddate}_5days.nc')

        t_init_all.append(input_constrained['temperature'][0, 1].loc[lev_t] - 273.15)  # the sencond (last) step of time
        t_clim_init_all.append(input_clim_constrained['temperature'][0, 1].loc[lev_t] - 273.15)
        z_init_all.append(input_constrained['geopotential'][0, 1].loc[lev_z] / 9.80665)
        z_clim_init_all.append(input_clim_constrained['geopotential'][0, 1].loc[lev_z] / 9.80665)

        t_constrained_all.append(predictions_constrained['temperature'].loc[:, 0, lev_t] - 273.15)  # select all times
        t_clim_constrained_all.append(predictions_clim_constrained['temperature'].loc[:, 0, lev_t] - 273.15)
        z_constrained_all.append(predictions_constrained['geopotential'].loc[:, 0, lev_z] / 9.80665)
        z_clim_constrained_all.append(predictions_clim_constrained['geopotential'].loc[:, 0, lev_z] / 9.80665)

    t_init_all = xr.concat(t_init_all, dim='initialization')
    t_clim_init_all = xr.concat(t_clim_init_all, dim='initialization')
    z_init_all = xr.concat(z_init_all, dim='initialization')
    z_clim_init_all = xr.concat(z_clim_init_all, dim='initialization')

    t_constrained_all = xr.concat(t_constrained_all, dim='initialization')
    t_clim_constrained_all = xr.concat(t_clim_constrained_all, dim='initialization')
    z_constrained_all = xr.concat(z_constrained_all, dim='initialization')
    z_clim_constrained_all = xr.concat(z_clim_constrained_all, dim='initialization')

    # # forecasts_time = np.datetime64(startdate) +  predictions_constrained['time'].values
    constrains_domain = predictions_constrained.attrs['constrains_domain']

    return (t_init_all, t_clim_init_all, z_init_all, z_clim_init_all,
            t_constrained_all, t_clim_constrained_all, z_constrained_all, z_clim_constrained_all,
            constrains_domain)



proj2 = ccrs.PlateCarree(central_longitude=180)

# colorbar_kw = dict(aspect=30, pad=0.02, shrink=0.9, ticklength=3, labelsize='large')  # 
colorbar_kw = dict(aspect=45, pad=0.08, shrink=0.5, ticklength=3)  # , labelsize='large'
levelz_pos = np.arange(50, 360, 50)  # np.arange(30, 325, 30)
levelz_neg = np.arange(-350, 0, 50)  # np.arange(-300, 0, 30)
lev_t = 850
lev_z = 500
target_color = 'gold' #'tab:green'
sig_color = 'orchid'
sig_kw = dict(pvalue=0.01, color=sig_color, alpha=0.8, method=1, size=1, hatches='///',)

vmax = 10 # 300
levels = np.linspace(-vmax, vmax, 21)
cmap = eplt.cmap_white_center(levels, cmap='RdBu_r')
g = 9.80665
unit = 'K'


startdates = ['2021-06-21T00','2021-06-21T06','2021-06-21T12','2021-06-21T18','2021-06-22T00']   
enddates = [(str(np.datetime64(s) + np.timedelta64(8, 'D'))[:13]) for s in startdates]

t_init_all, t_clim_init_all, z_init_all, z_clim_init_all, t_constrained_all, \
t_clim_constrained_all, z_constrained_all, z_clim_constrained_all, \
constrains_domain = get_data(startdates, enddates, lev_t, lev_z)

lonlim, latlim = [95, 255], [0, 80]
target_domain = [230, 250, 40, 60]   # domain to calculate regional average
lonw, lone, lats, latn = target_domain  


fig = plt.figure(figsize=(13, 6))  
gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig, hspace=0.4) # , wspace=0.3
proj = proj2

# First row
ax_top_left   = fig.add_subplot(gs[0, 0:2], projection=proj)
ax_top_center = fig.add_subplot(gs[0, 2:4], projection=proj)
ax_top_right  = fig.add_subplot(gs[0, 4:6], projection=proj)

# Second row
ax_bot_left  = fig.add_subplot(gs[1, 1:3], projection=proj)
ax_bot_right = fig.add_subplot(gs[1, 3:5], projection=proj)

axs = np.array([ax_top_left, ax_top_center, ax_top_right, ax_bot_left, ax_bot_right])

eplt.formats(axs, geo=True, abc='(a)', latloc=20, lonloc=30, latlim=latlim, lonlim=lonlim)

# Initial conditions
t_init_all = af.sellonlat(t_init_all, lonlim+latlim)
t_clim_init_all = af.sellonlat(t_clim_init_all, lonlim+latlim)
t_diff, t_p = composite_diff(t_clim_init_all, t_init_all)

z_init_all = af.sellonlat(z_init_all, lonlim+latlim)
z_clim_init_all = af.sellonlat(z_clim_init_all, lonlim+latlim)
z_diff, z_p = composite_diff(z_clim_init_all, z_init_all)

ax1 = axs[0]
eplt.formats(ax1, lonlim=lonlim, latlim=latlim, title='Initial condition')  # f'{startdate}'
# pic = eplt.contourf(ax1, z_diff, cmap=cmap, levels=levels, globe=True)
pic = eplt.contourf(ax1, t_diff, cmap=cmap, levels=levels, globe=True)
eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')
eplt.plt_sig(ax1, z_p, **sig_kw)
eplt.contour(ax1, z_diff, level=levelz_pos, clabel=False, fmt='%i', lw=0.7, color='k')
eplt.contour(ax1, z_diff, level=levelz_neg, clabel=False, fmt='%i', lw=0.7, color='k')

# Forecasts
steps = [8, 16, 24, 32]

for i, step in enumerate(steps):

    t_constrained = t_constrained_all[:, step-1]
    t_clim_constrained = t_clim_constrained_all[:, step-1]
    t_constrained = af.sellonlat(t_constrained, lonlim+latlim)
    t_clim_constrained = af.sellonlat(t_clim_constrained, lonlim+latlim)
    t_diff, t_p = composite_diff(t_clim_constrained, t_constrained, equal_var=False)
    t_diff = af.sellonlat(t_diff, lonlim+latlim)

    z_constrained = z_constrained_all[:, step-1]
    z_clim_constrained = z_clim_constrained_all[:, step-1]
    z_constrained = af.sellonlat(z_constrained, lonlim+latlim)
    z_clim_constrained = af.sellonlat(z_clim_constrained, lonlim+latlim)
    z_diff, z_p = composite_diff(z_clim_constrained, z_constrained, equal_var=False)
    z_diff = af.sellonlat(z_diff, lonlim+latlim)


    ax1 = axs[i+1]
    eplt.formats(ax1, lonlim=lonlim, latlim=latlim, title=f'Day {(i+1)*2}')  # 
    # pic = eplt.contourf(ax1, z_diff, cmap=cmap, levels=levels, globe=True)
    pic = eplt.contourf(ax1, t_diff, cmap=cmap, levels=levels, globe=True)
    eplt.contour(ax1, z_diff, level=levelz_pos, clabel=False, fmt='%i', lw=0.7, color='k')
    eplt.contour(ax1, z_diff, level=levelz_neg, clabel=False, fmt='%i', lw=0.7, color='k')
    eplt.plt_sig(ax1, z_p, **sig_kw)
    eplt.addpatch(ax1, target_domain, lw=1.5, ec=target_color)
    eplt.addpatch(ax1, constrains_domain, lw=1.5, ec='r')

t_diff_avg = af.weighted_avg(af.sellonlat(t_constrained-t_clim_constrained, target_domain)).values

eplt.addcolorbar(axs, pic, ticks=pic.levels[::2], label=f'T850 difference ({unit})', **colorbar_kw)  # For 2 columns

fig.savefig(dir_fig+'figure3.png', dpi=300, bbox_inches='tight')
fig.savefig(dir_fig+'figure3.pdf', dpi=300, bbox_inches='tight')
