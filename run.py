from time import perf_counter
import tensorrt as trt
import onnxruntime as ort
import pycuda.autoinit
import pycuda.driver as cuda
from loguru import logger
import numpy as np
from common.text.text_processing import get_text_processing
from common.text import cmudict

from box import Box
from pathlib import Path
from typing import Union
# from convert_models import convert_fastpitch
import torch

from torch.nn.utils.rnn import pad_sequence


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Reference:
#  https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py


def get_args(config_file: Union[Path, str]) -> Box:
    """
    Get configuration arguments

    Args:
        config_file (Union[Path, str]): Path to configuration file

    Returns:
        config (Box): Configuration arguments
    """
    return Box.from_yaml(filename=config_file)

config = get_args(Path("config.yaml"))

args: Box = config.inference
model_args: Box = config.model_config


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

onnxfile = "pretrained_models/hifigan.onnx"
# # Sanity check
# onnx_model = onnx.load(onnxfile)
# onnx.checker.check_model(onnx_model)
# del onnx_model

major, minor, patch = trt.__version__.split('.')

logger.info(f"TensorRT Version: {major}.{minor}.{patch}")
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #we have enabled the explicit Batch


# with (trt.Builder(TRT_LOGGER) as builder, 
#       builder.create_network() as network, 
#       trt.OnnxParser(network, TRT_LOGGER) as parser, 
#       builder.create_builder_config() as builder_config):
#     builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<< 31)

#     # if args.use_amp:
#     #     logger.info("Using FP16 Precision")
#     #     builder_config.set_flag(trt.BuilderFlag.FP16)

#     logger.info("Parsing ONNX file.")
#     with open(onnxfile, 'rb') as model:
#         if not parser.parse(model.read()):
#             for error in range(parser.num_errors):
#                 logger.error(parser.get_error(error))
#             logger.error("Failed to parse ONNX file.")
#             raise Exception("Failed to parse ONNX file.")

#     # Set dynamic shapes
#     logger.info("Setting optimization profile.")
#     profile = builder.create_optimization_profile()
#     # TODO: Set proper dynamic shapes
#     profile.set_shape("spec", (1, 80, 161), (4, 80, 261), (8, 80, 861))

#     builder_config.add_optimization_profile(profile)
#     builder_config.default_device_type = trt.DeviceType.GPU

#     logger.info("Building TensorRT engine. This may take a few minutes.")
#     serialized_engine = builder.build_serialized_network(network, builder_config)

#     with open('pretrained_models/hifigan.plan', 'wb') as f:
#         f.write(serialized_engine)
#         logger.info("TensorRT engine saved to pretrained_models/hifigan.plan")


with open("pretrained_models/hifigan.plan", "rb") as f:
    serialized_engine = f.read()
    logger.info("Engine loaded successfully")

engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    logger.info(f"Tensor:{tensor_name}, Shape:{engine.get_tensor_shape(tensor_name)}")


# class

def infer(spec: np.array):
    # https://github.com/NVIDIA/TensorRT/issues/4230
    # Actual shapes of the inputs
    input_shapes = spec.shape

    B = input_shapes[0]
    inputs = []
    outputs = []
    bindings = []
    context.set_input_shape("spec", spec.shape)
    # print(context.get_tensor_shape("audio"))
    context.infer_shapes()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        # print(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Check if it's an input or output tensor
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            shape = input_shapes  # Get the shape from the input shapes

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            inputs.append(HostDeviceMem(host_mem, device_mem))
            bindings.append(int(device_mem))

            np.copyto(inputs[-1].host, locals()[tensor_name].ravel())
        else:
            temp_shape = context.get_tensor_shape(tensor_name)
            
            # temp_shape = (1,)  # Placeholder, adjust if necessary
            size = trt.volume(temp_shape)
            # print(temp_shape, size, dtype)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            outputs.append(HostDeviceMem(host_mem, device_mem))
            bindings.append(int(device_mem))

    # Transfer inputs to device
    for i in range(len(inputs)):
        cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)

    # Set tensor address for each input/output
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    context.execute_async_v3(stream.handle)

    # Transfer predictions back
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)

    # Synchronize the stream
    stream.synchronize()

    return outputs[0].host

def cleanup():
    for input_mem in inputs:
        input_mem.device.free()  # Free device memory for each input
    for output_mem in outputs:
        output_mem.device.free()  # Free device memory for each output



# Run inference
mel = np.random.rand(2, 80, 361).astype(np.float32)
start_time = perf_counter()
output = infer(mel)
end_time = perf_counter()
# print time in milliseconds
print(f"Time taken:{(end_time - start_time) * 1000:.2f} ms")
print("Output shape:", output.shape)

print(output)

# Clean up memory after inference
cleanup()