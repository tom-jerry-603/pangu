import h5py

with h5py.File('input_data/predictions_6h.h5', 'r') as fin:
    with h5py.File('input_data/ready_6h.h5', 'w') as fout:
        if 'surface' not in fout:
            surface = fout.create_dataset('surface', shape=(40, 4, 721, 1440), dtype='float32', chunks=True)
        else:
            surface = fout["surface"]
        if 'upper' not in fout:
            upper = fout.create_dataset('upper', shape=(40, 5, 13, 721, 1440), dtype='float32', chunks=True)
        else:
            upper = fout["upper"]

        print(fin["mean_sea_level_pressure"][:].shape)
        surface[:, 0, :, :] = fin["mean_sea_level_pressure"][:, 0]
        surface[:, 1, :, :] = fin["10m_u_component_of_wind"][:, 0]
        surface[:, 2, :, :] = fin["10m_v_component_of_wind"][:, 0]
        surface[:, 3, :, :] = fin["2m_temperature"][:, 0]
        print(fin["geopotential"].shape)
        upper[:, 0, :, :, :] = fin["geopotential"][:, 0]
        upper[:, 1, :, :, :] = fin["specific_humidity"][:, 0]
        upper[:, 2, :, :, :] = fin["temperature"][:, 0]
        upper[:, 3, :, :, :] = fin["u_component_of_wind"][:, 0]
        upper[:, 4, :, :, :] = fin["v_component_of_wind"][:, 0]
