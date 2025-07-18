import h5py
import numpy as np

with h5py.File("input_data/predictions_6h.h5", "r") as f_origin:
    tp = f_origin["total_precipitation_6hr"][:]

tp = np.repeat(tp, repeats=6, axis=0)
print(tp.shape)

with h5py.File("output_data/out.h5", "r") as f_origin:
    t2m = f_origin["t2m"][:]

t2m = t2m[:, np.newaxis, :, :]

tp/=6

# Concatenate along axis=1
combined = np.concatenate([t2m, tp], axis=1)

with h5py.File("output_data/all.h5", "w") as f:
    f.create_dataset("forecast", data=combined)