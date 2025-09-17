import numpy as np
import xarray as xr


def sellonlat(arr, domain, nst=1):
    # opt: 0) transfer from -180~180 to 0~360; 1) do not transfer

    lonw, lone, lats, latn = domain

    if hasattr(arr, 'longitude'):
        lonname = 'longitude'
        lonmin = arr.longitude[0].values
    elif hasattr(arr, 'lon'):
        lonname = 'lon'
        lonmin = arr.lon[0].values  # 0 or -180
    else:
        raise ValueError('Unknown Longitude Name!')


    if hasattr(arr, 'latitude'):
        dlat = arr.latitude[0] - arr.latitude[1]
    elif hasattr(arr, 'lat'):
        dlat = arr.lat[0] - arr.lat[1]
    else:
        raise ValueError('Unknown Latitude Name!')

    if dlat < 0 and latn > lats:
        latn, lats = lats, latn

    if lone < lonw:
        raise ValueError('lone must >= lonw!')

    if (lone > 180 and lonw < 180) & (lonmin<0):
        # cross 180
        lone1 = lone - 360
        arr1 = arr.loc[..., latn:lats:nst, lonw:180]
        arr2 = arr.loc[..., latn:lats:nst, -180:lone1]
        arr3 = xr.concat((arr1, arr2), dim=lonname)
        arr3[lonname] = arr3[lonname].where(arr3[lonname]>=0, other=arr3[lonname]+360)  # 学习assign_coords的使用
        return arr3[..., ::nst]


    elif (lone > 0 and lonw < 0) & (lonmin>=0):
        # cross 0
        lonw1 = lonw + 360
        arr1 = arr.loc[..., latn:lats:nst, lonw1:360]
        arr2 = arr.loc[..., latn:lats:nst, 0:lone]
        arr3 = xr.concat((arr1, arr2), dim=lonname)
        arr3[lonname] = arr3[lonname].where(arr3[lonname]<=180, other=arr3[lonname]-360)
        return arr3[..., ::nst]

    else:
        if (lone > 180 and lonw >= 180) & (lonmin<0):
            lonw -= 360
            lone -= 360
        if (lone <= 0 and lonw < 0) & (lonmin>=0):
            lonw += 360
            lone += 360
        return arr.loc[..., latn:lats:nst, lonw:lone:nst]


def weighted_avg(arr:xr.DataArray):
    # the last two dimensions must be latitude and longitude
    return arr.weighted(np.cos(np.deg2rad(arr[arr.dims[-2]]))).mean(arr.dims[-2:])
