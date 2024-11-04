import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from loguru import logger
import onnxruntime as ort
from pathlib import Path 
from typing import Tuple

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()



class TTS:

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
    hifigan_planfile = "pretrained_models/hifigan.plan"
    fastpitch_onnxfile = "pretrained_models/fastpitch.onnx"
    hifigan_onnxfile = "pretrained_models/hifigan.onnx"

    options = ort.SessionOptions()
    options.enable_profiling=False

    def __init__(self):
        # Get total available GPU Memory
        # print(dir(cuda))
        self.cuda_context = cuda.Device(0).make_context()
        self.total_memory = cuda.Device(0).total_memory()
        total_memory_fraction = 0.6

        self.memory_pool = int(total_memory_fraction * self.total_memory)

        logger.info(f"Total Available GPU Memory: {self.memory_pool // 1024**3} GiB")

        # Allocate memory pools for each framework
        self.trt_memory = int(0.6 * self.memory_pool)
        self.onnx_memory = int(0.4 * self.memory_pool)

        logger.info(f"Allocating ~{self.trt_memory / 1024**3:.2f} GiB for TensorRT")
        logger.info(f"Allocating ~{self.onnx_memory / 1024**3:.2f} GiB for ONNX Runtime")

        self.cuda_context.push()

        # ====================================================================

        if not Path(self.hifigan_planfile ).exists():
            logger.info("HiFiGAN TensorRT Engine not found. Building engine.")
            self._build_engine()
        else:
            logger.info("HiFiGAN TensorRT Engine found. Loading engine.")
        self.runtime = trt.Runtime(self.TRT_LOGGER)

        # Load HifiGAN TensorRT Engine
        with open(self.hifigan_planfile , "rb") as f:
            serialized_engine = f.read()
            logger.info("HiFiGAN Engine loaded successfully")

        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)

        assert self.engine is not None, "Engine not loaded"
        self.trt_context = self.engine.create_execution_context()
        # Set workspace memory
        # Allocate memory for engine and binding buffers
        self.trt_buffer = cuda.mem_alloc(self.trt_memory)
        self.trt_context.set_optimization_profile_async(0, self.trt_buffer)
        self.stream = cuda.Stream()

        # ====================================================================

        logger.info("Creating ONNX Runtime Session for FastPitch")
        # self.onnx_buffer = cuda.mem_alloc(self.onnx_memory)
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        # self.options.gpu_mem_limit = self.onnx_memory
        self.ort_session = ort.InferenceSession( 
                    self.fastpitch_onnxfile,
                    sess_options=self.options,
                    providers=[('CUDAExecutionProvider', 
                                {
                                # Disable CUDA Graph Capture for GPU Partitioning
                                #  "enable_cuda_graph": '1', # Enable CUDA graph capture
                                 'device_id': 0, 
                                 'gpu_mem_limit': self.onnx_memory,
                                 'do_copy_in_default_stream': True,
                                 'arena_extend_strategy': 'kNextPowerOfTwo',})])
        
        # ====================================================================
        logger.info("TTS Server Initialized")

        
    def _build_engine(self):
        logger.info("Building HiFiGAN TensorRT Engine")
        with (trt.Builder(self.TRT_LOGGER) as builder, 
              builder.create_network() as network, 
              trt.OnnxParser(network, self.TRT_LOGGER) as parser, 
              builder.create_builder_config() as builder_config):
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<< 31)

            # if args.use_amp:
            #     logger.info("Using FP16 Precision")
            #     builder_config.set_flag(trt.BuilderFlag.FP16)

            logger.info("Parsing HiFiGAN tch ONNX file.")
            with open(self.hifigan_onnxfile, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    logger.error("Failed to parse HiFiGAN ONNX file.")
                    raise Exception("Failed to parse HiFiGAN ONNX file.")

            # Set dynamic shapes
            logger.info("Setting optimization profile.")
            profile = builder.create_optimization_profile()
            # TODO: Set proper dynamic shapes
            profile.set_shape("spec", (1, 80, 161), (4, 80, 261), (8, 80, 861))

            builder_config.add_optimization_profile(profile)
            builder_config.default_device_type = trt.DeviceType.GPU

            logger.info("Building HiFiGAN TensorRT engine. This may take a few minutes.")
            serialized_engine = builder.build_serialized_network(network, builder_config)

            with open('pretrained_models/hifigan.plan', 'wb') as f:
                f.write(serialized_engine)
                logger.info("HiFiGAN TensorRT engine saved to pretrained_models/hifigan.plan")

    def infer_fastpitch(self, encoded_text: str, pace:float = 1.0, pitch:float = 1.0):
        logger.info("Running FastPitch Inference")
        self.cuda_context.synchronize()
        # encoded_text = fastpitch.parse(text).detach().cpu().numpy()
        # print(encoded_text.shape)
        pace = np.array([[pace]], dtype=np.float32)
        pitch = np.array([[pitch]], dtype=np.float32)

        encoded_text = ort.OrtValue.ortvalue_from_numpy(encoded_text)
        pace = ort.OrtValue.ortvalue_from_numpy(pace)
        pitch = ort.OrtValue.ortvalue_from_numpy(pitch)

        logger.info(f"{len(self.ort_session.get_inputs())}")
        logger.info(f"{self.ort_session.get_outputs()[0].shape}")

        results = self.ort_session.run(None, input_feed= {"text": encoded_text, "pitch":pitch, "pace": pace})

        spec = results[0]

        logger.info(f"FastPitch Inference Complete. Shape: {spec.shape}")

        return spec

    def infer_hifigan(self, spec: np.array):
        logger.info("Running HiFiGAN Inference")
        self.cuda_context.synchronize()
        input_shapes = spec.shape

        B = input_shapes[0]
        inputs = []
        outputs = []
        bindings = []
        self.trt_context.set_input_shape("spec", spec.shape)
        self.trt_context.infer_shapes()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            # print(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Check if it's an input or output tensor
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                shape = input_shapes  # Get the shape from the input shapes

                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                inputs.append(HostDeviceMem(host_mem, device_mem))
                bindings.append(int(device_mem))

                np.copyto(inputs[-1].host, locals()[tensor_name].ravel())
            else:
                temp_shape = self.trt_context.get_tensor_shape(tensor_name)
                
                # temp_shape = (1,)  # Placeholder, adjust if necessary
                size = trt.volume(temp_shape)
                # print(temp_shape, size, dtype)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                outputs.append(HostDeviceMem(host_mem, device_mem))
                bindings.append(int(device_mem))

        logger.info(f"Transferring {len(inputs)} inputs to GPU")
        # Transfer inputs to device
        for i in range(len(inputs)):
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, self.stream)

        # Set tensor address for each input/output
        for i in range(self.engine.num_io_tensors):
            self.trt_context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])

        logger.info("Executing HiFiGAN Inference")
        self.trt_context.execute_async_v3(self.stream.handle)

        logger.info("Transferring output back to host")
        # Transfer predictions back
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return outputs[0].host

    def run_pipeline(self, inputs:dict):
        # Step 1: ONNX Inference
        # Bind input to ONNX
        io_binding = self.ort_session.io_binding()

        input_names = [input.name for input in self.ort_session.get_inputs()]
        output_names = [output.name for output in self.ort_session.get_outputs()]

        
        for input_name in input_names:
            data = inputs[input_name]
            # Allocate and copy input data to GPU
            d_input = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(d_input, data)

            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=data.shape,
                buffer_ptr=int(d_input)
            )
        
        # for output_name in output_names:


        # Get ONNX output shape and allocate intermediate buffer
        onnx_output_shape = self.ort_session.get_outputs()[0].shape
        # if not self.intermediate_buffer:
        #     intermediate_size = np.prod(onnx_output_shape) * np.dtype(np.float32).itemsize
        #     self.intermediate_buffer = cuda.mem_alloc(intermediate_size)
        
        # Bind ONNX output to intermediate buffer
        io_binding.bind_output(
            name=output_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=onnx_output_shape,
            buffer_ptr=int(self.intermediate_buffer)
        )
        
        # Run ONNX inference
        self.ort_session.run_with_iobinding(io_binding)
        self.cuda_context.synchronize()
        
        # Free input buffer
        d_input.free()
        
        # Step 2: TensorRT Inference
        # Get TensorRT output shape and allocate output buffer
        output_idx = 1  # Assuming single output
        trt_output_shape = self.trt_context.engine.get_binding_shape(output_idx)
        h_output = cuda.pagelocked_empty(tuple(trt_output_shape), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Run TensorRT inference using intermediate buffer as input
        bindings = [int(self.intermediate_buffer), int(d_output)]
        self.trt_context.execute_v2(bindings)
        
        # Copy final output back to host
        cuda.memcpy_dtoh(h_output, d_output)
        self.context.synchronize()
        
        # Free output buffer
        d_output.free()
        
        return h_output
    

    def cleanup(self):
        if self.trt_buffer:
            self.trt_buffer.free()
        # if self.onnx_buffer:
        #     self.onnx_buffer.free()
        logger.info("Closing CUDA Context")
        self.cuda_context.pop()


if __name__ == "__main__":
    tts = TTS()
    text = "Hello, how are you doing today?"
    text = np.random.randint(0, 100, (1, 122))
    # print(text.shape)
    spec = tts.infer_fastpitch(text)
    # print(spec.shape)
    audio = tts.infer_hifigan(spec)
    print(audio.shape)
    tts.cleanup()

    # with GPUPartitionManager() as gpu_manager:
    #     # Load models
    #     gpu_manager.load_tensorrt_engine("pretrained_models/hifigan.plan")
    #     gpu_manager.load_onnx_session("pretrained_models/fastpitch.onnx")
        
    #     # Prepare input
    #     input_data = np.random.rand(1, 122).astype(np.float32)
        
    #     # Run inference
    #     trt_output, onnx_output = gpu_manager.run_inference(input_data)