import cdsapi
import xarray as xr
import numpy as np
import os

# Delete a file
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File deleted.")
    else:
        print("File does not exist.")

c = cdsapi.Client()


def input_surface(year, month, day):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'mean_sea_level_pressure', '10m_u_component_of_wind',
                '10m_v_component_of_wind', '2m_temperature'
            ],
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': f'{day:02d}',
            'time': '00:00',
            'format': 'netcdf',
        },
        'input_data/surface.nc')

    ds = xr.open_dataset('input_data/surface.nc')
    ds = ds.sel(valid_time=f'{year}-{month:02d}-{day:02d}T00:00')
    variables = ['msl', 'u10', 'v10', 't2m']
    data = np.stack([ds[var].values.squeeze() for var in variables])  # shape: (4, 721, 1440)
    np.save('input_data/input_surface.npy', data)
    ds.close()
    delete_file('input_data/surface.nc')

def input_upper(year, month, day):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind'
            ],
            'pressure_level': [
                '1000', '925', '850', '700', '600', '500', '400',
                '300', '250', '200', '150', '100', '50'
            ],
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': f'{day:02d}',
            'time': '00:00',
            'format': 'netcdf',
        },
        'input_data/upper.nc'
    )

    ds = xr.open_dataset("input_data/upper.nc")

    pressure_levels = [
        1000, 925, 850, 700, 600, 500, 400,
        300, 250, 200, 150, 100, 50
    ]

    ds = ds.sel(
        valid_time=f"{year}-{month:02d}-{day:02d}T00:00",
        pressure_level=pressure_levels
    )

    # Variables in correct order: Z, Q, T, U, V
    Z = ds['z'].values # convert geopotential to geopotential height (m)
    Q = ds['q'].values
    T = ds['t'].values
    U = ds['u'].values
    V = ds['v'].values

    # Stack into shape: (5, 13, 721, 1440)
    data = np.stack([Z, Q, T, U, V], axis=0)

    # Save
    np.save("input_data/input_upper.npy", data)
    ds.close()
    delete_file('input_data/upper.nc')

if __name__ == "__main__":
    year, month, day = 2025, 7, 9
    input_surface(year, month, day)
    input_upper(year, month, day)