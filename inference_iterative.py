import os
import numpy as np
import onnx
import onnxruntime as ort
import h5py

# Optional: clear GPU memory if PyTorch is available
try:
    import torch
    torch.cuda.empty_cache()
except ImportError:
    pass

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
os.makedirs(output_data_dir, exist_ok=True)

# ONNXRuntime Session Options
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 1

cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}
ort_session_3 = ort.InferenceSession('pangu_weather_3.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

def predict_3h(): 
    with h5py.File("input_data/ready_6h.h5", 'r') as fin:
        in_len = fin['upper'].shape[0]
        with h5py.File("input_data/ready_3h.h5", 'w') as f:
            if 'surface' not in f:
                surface = f.create_dataset('surface', shape=(in_len * 2, 4, 721, 1440), dtype='float32', chunks=True)
            else:
                surface = f["surface"]
            if 'upper' not in f:
                upper = f.create_dataset('upper', shape=(in_len * 2, 5, 13, 721, 1440), dtype='float32', chunks=True)
            else:
                upper = f["upper"]
            
            for i in range(in_len):
                input_3, input_surface_3 = fin["upper"][i][:, ::-1, ::-1], fin["surface"][i][:, ::-1]
                surface[i * 2] = input_surface_3
                upper[i * 2] = input_3
                # print(input_3, input_surface_3)
                output, output_surface = ort_session_3.run(None, {
                    'input': input_3, 'input_surface': input_surface_3
                })
                # print(output, output_surface)
                surface[i * 2 + 1] = output_surface
                upper[i * 2 + 1] = output

                print(f"timestamp:3h Step {i * 2} / {in_len * 2} completed!", end='\r')

    print("Inference and writing completed.")

def predict_1h():
    ort_session_1 = ort.InferenceSession('pangu_weather_1.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
    with h5py.File('input_data/ready_3h.h5', 'r') as fin:
        with h5py.File('output_data/out.h5', 'w') as f:
            if 't2m' in f:
                t2m = f['t2m']
            else:
                t2m = f.create_dataset('t2m', shape=(240, 721, 1440), dtype='float32', chunks=True)
            for i in range(80):
                input_1, input_surface_1 = fin["upper"][i], fin["surface"][i]
                
                t2m[i * 3] = input_surface_1[3]
                for j in range(2):
                    output, output_surface = ort_session_1.run(None, {
                        'input': input_1, 'input_surface': input_surface_1
                    })
                    t2m[i * 3 + j + 1] = output_surface[3]
                    input_1, input_surface_1 = output, output_surface

                print(f"timestamp:1h Step {(i + 1) * 3} / 240 completed!", end='\r')

    print("Inference and writing completed.")
if __name__ == "__main__":
  predict_3h()
  predict_1h()