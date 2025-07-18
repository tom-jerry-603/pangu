import numpy as np
import h5py

# Assuming A and B are your two arrays
# Shape: (312, 721, 1440)
with h5py.File("output_data/out.h5", "r") as pred:
    A = pred["t2m"][:120]
A = A[:120, :, :]
B = np.load("output_data/real_data/t2m.npy").astype(np.float32)[:120]

print(A[:, 120, 0], B[:, 120, 0])

rmse = np.sqrt(np.mean((A - B) ** 2))
print("RMSE:", rmse)