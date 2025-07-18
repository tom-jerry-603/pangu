import gdown

file_ids = ["1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S", "1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt", "1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4", "1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP"]
outputs = ["pangu_weather_1.onnx", "pangu_weather_3.onnx", "pangu_weather_6.onnx", "pangu_weather_24.onnx"]

for file_id, output in zip(file_ids, outputs):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)