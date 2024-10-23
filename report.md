
# Pipelining ONNX and TensorRT Models

In this report, we shall discuss and document the process of pipelining two models, one in ONNX and the other in TensorRT. The models in question are FastPitch and HiFi-GAN respectively. The FastPitch model is a text-to-speech model that generates mel spectrograms from text. The HiFi-GAN model is a vocoder that converts the mel spectrograms to audio waveforms. Naturally, they can be pipelined to generate audio from text.


## Pipeline Description
The [FastPitch](https://arxiv.org/abs/2006.06873) model is converted to the TensorRT format. This is done by first converting the PyTorch model to TorchScript. A JIT-able pytorch code for FastPitch model can be obtained from this [GitHub repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch). With the newer PyTorch versions (> 2.0), converting to TensorRT is quite easy. PyTorch, now comes with a compiler that supports a variety of backends, including `torch_tensorrt`.


The [HiFi-GAN](https://arxiv.org/abs/2010.05646) model is converted to the ONNX format. Although, the newer versions of PyTorch provides a new JIT compiler - Dynamo, that can be used to export to ONNX, it is tricky to do so. The reason being that it does not support exporting models with named input and output tensors. This can be pose a problem when pipelining the models, as it is easier to bind the tensors based on their names. As such, we have to resort to the legacy api - `torch.onnx.export`.


A reason for the above choice is that the FastPitch model, with the current code, cannot be directly converted to ONNX model. In retrospect, I could have used the model from NVIDIA's Nemo library. They do provide the [code](https://github.com/NVIDIA/NeMo/blob/f45f56bb3730939f43ef2a8656bf8075d615f361/nemo/collections/tts/models/fastpitch.py) for the FastPitch model that can be exported to ONNX.


The two models, after their respective conversions, can be chained together to for an end-to-end text-to-speech system. The FastPitch TensorRT model can as is - calling the infer function with the encoded input text. The HiFi-GAN ONNX model can be used with the `onnxruntime` - creating an inference session. I have used both TensorRT and CUDA as execution providers for faster inference on the GPU. Since the output of the FastPitch model still remains on the same GPU, the ONNX Inference session can be directly bound to the output tensor by name (Hence, we used the legacy `torch.onnx.export` api). Similarly, the output of the ONNX model can be exported to the host memory by name. 


For the servering the above TTS model, we shall use [LitServe](https://github.com/Lightning-AI/LitServe) framework. The client library is a simple python script that sends a text to the server and receives the audio waveform via the `requests` library. These are available in the `client.py` and `tts_server.py` files respectively.

## Results


GPU Specs
```
Model:           NVIDIA GeForce RTX 3070 Laptop GPU
IRQ:             191
GPU UUID:        GPU-d3eb01d5-5c82-d972-bcbb-72c3f67c19f3
Video BIOS:      94.04.43.00.8d
Bus Type:        PCIe
DMA Size:        47 bits
DMA Mask:        0x7fffffffffff
Bus Location:    0000:01:00.0
Device Minor:    0
GPU Firmware:    560.35.03
GPU Excluded:    No
```

### Latency
FastPitch (TensorRT) + HiFi-GAN (ONNX) (without denoising)

| Batch size | Precision | Avg Latency(s) | Latency 90% (s) | Latency 95% (s) | Latency 99% (s) | Throughput (samples/sec) | Avg RTF |
| ---------- | --------- | -------------- | --------------- | --------------- | --------------- | ------------------------ | ------- |
| 1          | FP16      | 0.0126         | 0.0137          | 0.0139          | 0.0143          | 13,452,083               | 610.07  |
| 4          | FP16      | 0.0313         | 0.0340          | 0.0351          | 0.0358          | 21,625,047               | 245.81  |
| 8          | FP16      | 0.0693         | 0.0733          | 0.0737          | 0.0743          | 19,750,759               | 110.73  |
| 1          | TF32      | 0.0215         | 0.0830          | 0.0948          | 0.1178          | 5,436,562                | 156.77  |
| 4          | TF32      | 0.1670         | 0.1710          | 0.1755          | 0.1755          | 5,829,018                | 98.400  |
| 8          | TF32      | 0.3250         | 0.3250          | 0.3310          | 0.3310          | 5,253,920                | 64.451  |


### CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 | Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation                      |
 | -------- | --------------- | ----- | -------- | -------- | -------- | -------- | ----------- | ------------------------------ |
 | 81.0     | 728394338       | 28708 | 25372.5  | 928.0    | 352      | 4660889  | 124756.4    | [CUDA memset]                  |
 | 8.1      | 73051150        | 994   | 73492.1  | 992.5    | 352      | 9054320  | 351905.5    | [CUDA memcpy Host-to-Device]   |
 | 6.3      | 56843830        | 817   | 69576.3  | 1536.0   | 929      | 8871373  | 378621.2    | [CUDA memcpy Device-to-Host]   |
 | 4.5      | 40774609        | 2641  | 15439.1  | 2400.0   | 1119     | 142433   | 33589.2     | [CUDA memcpy Device-to-Device] |

### CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 | Total (MB) | Count | Avg (MB) | Med (MB) | Min (MB) | Max (MB) | StdDev (MB) | Operation                      |
 | ---------- | ----- | -------- | -------- | -------- | -------- | ----------- | ------------------------------ |
 | 261298.862 | 28708 | 9.102    | 0.001    | 0.000    | 1386.217 | 44.582      | [CUDA memset]                  |
 | 7199.670   | 2641  | 2.726    | 0.677    | 0.000    | 21.660   | 5.833       | [CUDA memcpy Device-to-Device] |
 | 470.632    | 994   | 0.473    | 0.001    | 0.000    | 55.724   | 2.188       | [CUDA memcpy Host-to-Device]   |
 | 353.711    | 817   | 0.433    | 0.001    | 0.000    | 55.724   | 2.342       | [CUDA memcpy Device-to-Host]   |

### CUDA API Summary (cuda_api_sum):

 | Time (%) | Total Time (ns) | Num Calls | Avg (ns)  | Med (ns)  | Min (ns) | Max (ns)  | StdDev (ns) | Name                           |
 | -------- | --------------- | --------- | --------- | --------- | -------- | --------- | ----------- | ------------------------------ |
 | 73.3     | 24286748082     | 20848     | 1164943.8 | 141392.0  | 3692     | 269804738 | 5603330.9   | cudaEventSynchronize           |
 | 20.2     | 6685831574      | 16434     | 406829.2  | 4551.5    | 442      | 336004787 | 11289600.8  | cudaStreamSynchronize          |
 | 1.6      | 535101963       | 83228     | 6429.4    | 2682.0    | 1849     | 41892449  | 236400.7    | cudaLaunchKernel               |
 | 1.4      | 471922021       | 7412      | 63670.0   | 43189.5   | 19381    | 2217869   | 77680.8     | cuModuleLoadFatBinary          |
 | 0.7      | 230023573       | 7556      | 30442.5   | 15045.5   | 4903     | 18952438  | 232451.9    | cuModuleUnload                 |
 | 0.5      | 181448417       | 38615     | 4698.9    | 2607.0    | 57       | 132898    | 7514.5      | cudaMemsetAsync                |
 | 0.5      | 157154635       | 20960     | 7497.8    | 345.0     | 159      | 2335126   | 61713.5     | cudaFree                       |
 | 0.4      | 143295279       | 4943      | 28989.5   | 703.0     | 53       | 9013900   | 223134.8    | cudaMemcpyAsync                |
 | 0.3      | 89241057        | 1617      | 55189.3   | 3171.0    | 2176     | 68056671  | 1724500.9   | cudaLaunchKernelExC_v11060     |
 | 0.3      | 83316874        | 17344     | 4803.8    | 2575.0    | 1692     | 2289881   | 18117.4     | cuLaunchKernelEx               |
 | 0.2      | 75993456        | 44186     | 1719.9    | 1286.0    | 903      | 42291     | 1563.7      | cudaEventRecord                |
 | 0.1      | 36179962        | 2878      | 12571.2   | 4513.5    | 1588     | 1953013   | 54104.0     | cudaMalloc                     |
 | 0.1      | 27906539        | 1196      | 23333.2   | 12468.0   | 5402     | 8936530   | 260746.7    | cudaMemGetInfo                 |
 | 0.1      | 27052812        | 152       | 177979.0  | 65813.5   | 24963    | 4421675   | 571183.9    | cuModuleLoadData               |
 | 0.1      | 24503113        | 390       | 62828.5   | 52249.5   | 1514     | 6249922   | 318624.0    | cudaMallocAsync_v11020         |
 | 0.1      | 18473611        | 14        | 1319543.6 | 1156442.0 | 368619   | 2838011   | 866344.0    | cuLibraryLoadData              |
 | 0.0      | 11237499        | 43        | 261337.2  | 68273.0   | 25650    | 2094975   | 474821.0    | cuMemUnmap                     |
 | 0.0      | 10277714        | 3022      | 3401.0    | 2530.0    | 1679     | 33203     | 2542.6      | cuLaunchKernel                 |
 | 0.0      | 9312897         | 2153      | 4325.5    | 3256.0    | 2633     | 67262     | 3731.3      | cudaMemcpy2DAsync              |
 | 0.0      | 5492428         | 44        | 124827.9  | 29139.0   | 12837    | 2340087   | 359871.9    | cuLibraryUnload                |
 | 0.0      | 5041023         | 866       | 5821.0    | 532.5     | 255      | 3291075   | 114913.4    | cuKernelGetFunction            |
 | 0.0      | 4502778         | 43        | 104715.8  | 48859.0   | 30929    | 1594856   | 237725.6    | cuMemSetAccess                 |
 | 0.0      | 3946854         | 286       | 13800.2   | 12166.0   | 5084     | 64753     | 9652.5      | cuMemHostAlloc                 |
 | 0.0      | 3547636         | 708       | 5010.8    | 4391.5    | 1948     | 46425     | 3632.4      | cudaStreamCreate               |
 | 0.0      | 2447470         | 286       | 8557.6    | 7790.5    | 5134     | 38282     | 4010.4      | cuMemFreeHost                  |
 | 0.0      | 2225042         | 741       | 3002.8    | 2601.0    | 1507     | 98356     | 3977.2      | cudaStreamDestroy              |
 | 0.0      | 1774156         | 43        | 41259.4   | 31382.0   | 13808    | 145370    | 26233.2     | cuMemCreate                    |
 | 0.0      | 1694583         | 4578      | 370.2     | 337.0     | 151      | 18465     | 452.7       | cudaStreamIsCapturing_v10000   |
 | 0.0      | 1588000         | 143       | 11104.9   | 3447.0    | 1455     | 985202    | 82115.7     | cuStreamDestroy_v2             |
 | 0.0      | 1583205         | 143       | 11071.4   | 6858.0    | 2556     | 212775    | 23628.4     | cuStreamCreate                 |
 | 0.0      | 1435717         | 2403      | 597.5     | 383.0     | 199      | 19120     | 676.7       | cudaEventCreateWithFlags       |
 | 0.0      | 984049          | 2384      | 412.8     | 299.0     | 188      | 19477     | 538.7       | cudaEventDestroy               |
 | 0.0      | 910664          | 43        | 21178.2   | 11921.0   | 10371    | 195106    | 32220.5     | cuMemRelease                   |
 | 0.0      | 674233          | 4         | 168558.3  | 20951.0   | 12260    | 620071    | 301097.6    | cudaHostAlloc                  |
 | 0.0      | 649103          | 49        | 13247.0   | 1298.0    | 971      | 334500    | 51017.0     | cudaStreamCreateWithFlags      |
 | 0.0      | 593289          | 390       | 1521.3    | 1607.5    | 830      | 24379     | 1263.6      | cudaFreeAsync_v11020           |
 | 0.0      | 538380          | 715       | 753.0     | 679.0     | 165      | 16783     | 831.1       | cuEventCreate                  |
 | 0.0      | 304559          | 2043      | 149.1     | 115.0     | 55       | 2917      | 123.9       | cuGetProcAddress_v2            |
 | 0.0      | 263157          | 715       | 368.1     | 383.0     | 122      | 2074      | 231.8       | cuEventDestroy_v2              |
 | 0.0      | 206772          | 43        | 4808.7    | 3489.0    | 1358     | 17544     | 3211.6      | cuMemMap                       |
 | 0.0      | 81933           | 477       | 171.8     | 162.0     | 98       | 765       | 59.1        | cuModuleGetLoadingMode         |
 | 0.0      | 78039           | 10        | 7803.9    | 2234.0    | 858      | 34880     | 10799.3     | cudaDeviceSynchronize          |
 | 0.0      | 68841           | 286       | 240.7     | 208.0     | 60       | 1253      | 172.5       | cuCtxSetCurrent                |
 | 0.0      | 64714           | 4         | 16178.5   | 15284.0   | 12733    | 21413     | 4128.1      | cuMemAddressReserve            |
 | 0.0      | 30744           | 2         | 15372.0   | 15372.0   | 12607    | 18137     | 3910.3      | cudaFreeHost                   |
 | 0.0      | 27518           | 4         | 6879.5    | 5584.5    | 1965     | 14384     | 5348.1      | cuMemAddressFree               |
 | 0.0      | 14661           | 21        | 698.1     | 587.0     | 332      | 1785      | 364.9       | cuLibraryGetKernel             |
 | 0.0      | 9443            | 6         | 1573.8    | 1421.0    | 1053     | 2289      | 524.7       | cuInit                         |
 | 0.0      | 1246            | 2         | 623.0     | 623.0     | 201      | 1045      | 596.8       | cudaGetDriverEntryPoint_v11030 |
 | 0.0      | 972             | 4         | 243.0     | 219.0     | 131      | 403       | 114.8       | cuMemGetAllocationGranularity  |


The memory snapshot file is also available under `assets/gpu_mem_snapshot.pickle` and can be visualized using the [PyTorch memory vix](https://pytorch.org/memory_viz).

### Issues Faced

The above FastPitch + HiFiGAN pipeline model is also available as an example from NVIDIA, and can be found [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch). However, I couldn't get the Triton Inference Server to work properly - as the `model.py` is missing a few functions. 

### Future Improvements
To improve the latency, the following can be done
1. Improve the data transfer between the models with zero-copy.
2. Create multiply CUDA streams to process the data in parallel.
3. Use Triton's dynamic batching based on the load.
4. FastPitch model uses the traditional MultiHeadAttention. This can be experimented with newer/faster transformer architectures, but requires re-training the model.

## References
1. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_e2e_fastpitchhifigan
2. https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1150/user-guide/docs/install.html#:~:text=The%20Triton%20Inference%20Server%20is,Docker%20and%20nvidia%2Ddocker%20installed.
3. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/export.html
4. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/checkpoints.html
5. https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch

