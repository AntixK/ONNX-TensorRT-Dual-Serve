## Multi-Framework Model Serving Optimization

**Background**: We need to efficiently serve both ONNX and TensorRT models on the same GPU infrastructure. Currently, running both frameworks in the same process leads to CUDA context conflicts and memory access issues.

**Assignment**: Develop and optimize a serving solution that enables concurrent execution of ONNX and TensorRT models on shared GPU resources.

### Requirements:

#### Implementation:
- Develop a server implementation that can run both ONNX and TensorRT models
- Or use Lightning Server or a similar framework for process and batch management
- Support batching of incoming requests
- Streaming is not needed

#### Performance Analysis:
- Compare your implementation's latency against using the models via NVIDIA - Triton Server
- Profile both solutions using nsys and nvprof
- Document the following metrics:
  - Inference latency (p50, p95, p99)
  - Throughput (RTFX)
  - GPU memory utilization
  - CUDA kernel execution times
  - Memory transfer overhead
  
#### Technical Report:
- Identify and analyze performance bottlenecks
- Document your architecture decisions
- Provide recommendations for further optimization

#### Models to use:
- Take any Fastpitch + HiFiGAN TTS model from Nvidia NGC
- Convert Fastpitch to either ONNX or TRT. 
- Convert HiFiGAN to the other implementation so as to string them together as one e2e system on a single GPU

#### Deliverables:

- Source code as a Git repo with documentation
- Performance comparison results
- Profiling data and analysis
- Technical report
