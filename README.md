# ONNX-TensorRT-Dual-Serve

## Requirements

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch 24.08-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
- Atleast 1 NVIDIA GPU with one of the following architectures:
  - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)


## Usage
### Download Assets

This includes the following
1. FastPitch (Trained on LJSpeech dataset)
2. HiFiGAN (trained on LJSpeech dataset)
3. cmudict 

```
bash downoad_assets.sh
```

### Launch Server and Client

- Build the docker image

```
make build-dev-env
```

- Launch the container
```
make run-dev-env
```
- Inside the container, open a terminal and launch the server
```
python tts_server.py
```
- In another terminal within the container, launch the client
```
python client.py
```


### Profile using NVIDIA's Nsys

```
nsys stats --force-export=true --report cuda_gpu_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report cuda_api_sum --format csv,column --output .,- report2.nsys-rep
```

