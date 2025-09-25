import sys
sys.path.append('../utils/')
import cdsapi
import pandas as pd
import utils


def output_format(fmt):
    if fmt=='netcdf':
        return fmt, 'nc'
    elif fmt=='grib':
        return fmt, fmt
    else:
        raise ValueError('Wrong format!')


def down_sl(date1, date2, path, fmt):
    c = cdsapi.Client()

    fmt1, fmt2 = output_format(fmt)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':fmt1,
            'variable':[
                '10m_u_component_of_wind', '10m_v_component_of_wind',
                '2m_temperature', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation'
            ],
            'date':f'{date1}/{date2}',
            'time':['00:00', '06:00', '12:00', '18:00'],
            'grid':[1.0, 1.0],
        },
        path+f'ERA5.{date1}.{ndays}days.single_level.{fmt2}')


def down_precip(date1, date2, path, fmt):
    c = cdsapi.Client()

    date1_before = pd.to_datetime(date1) - pd.Timedelta(days=1)
    date1_before = date1_before.strftime('%Y-%m-%d')

    fmt1, fmt2 = output_format(fmt)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':fmt1,
            'variable':['total_precipitation'],
            'date':f'{date1_before}/{date2}',
            'time':'00/to/23/by/1',
            'grid':[1.0, 1.0],
        },
        path+f'ERA5.{date1}.{ndays}days.total_precipitation.{fmt2}')


def down_invariants(path, fmt):
    c = cdsapi.Client()

    fmt1, fmt2 = output_format(fmt)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':fmt1,
            'variable':['land_sea_mask', 'geopotential'],
            'date':'2024-01-01',
            'time':['00:00'],
            'grid':[1.0, 1.0],
        },
        path+f'ERA5.invariants.{fmt2}')


def down_pl(date1, date2, path, fmt):
    c = cdsapi.Client()

    fmt1, fmt2 = output_format(fmt)

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type':'reanalysis',
            'format':fmt1,
            'variable':[
                'temperature', 'geopotential',
                'u_component_of_wind', 'v_component_of_wind',
                'vertical_velocity', 'specific_humidity',
            ],
            'pressure_level': [
                '50', '100', '150', '200', '250',
                '300', '400', '500', '600', '700',
                '850', '925', '1000'],
            'date':f'{date1}/{date2}',
            'time':['00:00', '06:00', '12:00', '18:00'],
            'grid':[1.0, 1.0],
        },
        path+f'ERA5.{date1}.{ndays}days.pressure_levels.{fmt2}')


def main(date1, date2, path, fmt):

    down_sl(date1, date2, path, 'grib')
    down_precip(date1, date2, path, fmt)
    down_invariants(path, fmt)
    down_pl(date1, date2, path, fmt)


if __name__ == '__main__':

    ########## Modify #############
    startdate = '2016-03-02'
    ndays = 14
    ###############################

    enddate = utils.get_enddate(startdate, ndays)

    dir_down = '/path/to/download/directory/'
    fmt = 'grib'

    main(startdate, enddate, dir_down, fmt)
