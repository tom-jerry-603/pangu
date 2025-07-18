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
                '2m_temperature'
            ],
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': [f'{(day + d):02d}' for d in range(1, 14)],
            'time': [f'{t:02d}:00' for t in range(24)],
            'format': 'netcdf',
        },
        'output_data/real_data/surface.nc')

    ds = xr.open_dataset('output_data/real_data/surface.nc')
    data = ds['t2m'].values # shape: (1, 721, 1440)
    np.save('output_data/real_data/t2m.npy', data)
    ds.close()
    delete_file('output_data/real_data/surface.nc')

if __name__ == "__main__":
    year, month, day = 2024, 7, 8
    input_surface(year, month, day)