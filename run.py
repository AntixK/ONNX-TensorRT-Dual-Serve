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
from convert_models import convert_fastpitch
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

cmudict.initialize(args.cmudict_path, args.heteronyms_path)

tp = get_text_processing(args.symbol_set,
                         args.text_cleaners,
                         args.p_arpabet)

text = "His disappearance gave color and substance to evil reports already in circulation that the will and conveyance above referred to"
encoded_text = np.asarray(tp.encode_text(text)).reshape(1, -1)

# encoded_text = [torch.LongTensor(tp.encode_text(text))]
# encoded_text = pad_sequence(encoded_text, batch_first=True)

# print(encoded_text.shape)

encoded_text = np.random.rand(2, 122).astype(np.int32)
# encoded_text = np.array([tp.encode_text(text)], dtype=np.int32)
pace = np.array(1.0, dtype=np.float32)

print(encoded_text.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
convert_fastpitch(model_args, args, device)

# Create ONNXRuntime Session for FastPitch
options = ort.SessionOptions()
options.enable_profiling=False
# options.log_severity=0# options.log_verbosity_level0


ort_session = ort.InferenceSession( 'pretrained_models/fastpitch.onnx',
    sess_options=options,
    providers=[ 'CUDAExecutionProvider'])

# print(dir(ort_session))
# print(ort_session.get_inputs())
# print(ort_session.get_outputs())
io_binding = ort_session.io_binding()

encoded_text = ort.OrtValue.ortvalue_from_numpy(encoded_text)
pace = ort.OrtValue.ortvalue_from_numpy(pace)

# ortvalue.device_name()  # 'cpu'
# ortvalue.shape()        # shape of the numpy array X
# ortvalue.data_type()    # 'tensor(float)'
# ortvalue.is_tensor()    # 'True'
# np.array_equal(ortvalue.numpy(), X)  # 'True'

results = ort_session.run(["mel"], {"text": encoded_text, "pace": pace})

print(results)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

with open("pretrained_models/hifigan.plan", "rb") as f:
    serialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    logger.info(f"Tensor:{tensor_name}, Shape:{engine.get_tensor_shape(tensor_name)}")


# class

def infer(mel: np.array):
    # https://github.com/NVIDIA/TensorRT/issues/4230
    # Actual shapes of the inputs
    input_shapes = mel.shape

    B = input_shapes[0]
    inputs = []
    outputs = []
    bindings = []
    context.set_input_shape("mel", mel.shape)

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
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
            temp_shape = (B, *engine.get_tensor_shape(tensor_name)[1:])
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
mel = np.random.rand(2, 80, 661).astype(np.float32)
audio = np.random.rand(1, 2, 16).astype(np.float32)
start_time = perf_counter()
output = infer(mel)
end_time = perf_counter()
# print time in milliseconds
print(f"Time taken:{(end_time - start_time) * 1000:.2f} ms")
print("Output shape:", output.shape)

print(output)

# Clean up memory after inference
cleanup()
