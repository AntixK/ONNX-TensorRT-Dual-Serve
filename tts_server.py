import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from loguru import logger
import onnxruntime as ort

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
    planfile = "pretrained_models/hifigan.plan"
    onnxfile = "pretrained_models/fastpitch.onnx"

    options = ort.SessionOptions()
    options.enable_profiling=False

    def __init__(self):
        self.runtime = trt.Runtime(self.TRT_LOGGER)

        # Load HifiGAN TensorRT Engine
        with open(self.onnxfile, "rb") as f:
            serialized_engine = f.read()
            logger.info("Engine loaded successfully")

        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.ort_session = ort.InferenceSession( 
                    self.onnxfile,
                    sess_options=self.options,
                    providers=[  'CUDAExecutionProvider'])

    def infer_fastpitch(self, encoded_text: str, pace:float = 1.0, pitch:float = 1.0):
        # encoded_text = fastpitch.parse(text).detach().cpu().numpy()
        # print(encoded_text.shape)
        pace = np.array([[pace]], dtype=np.float32)
        pitch = np.array([[pitch]], dtype=np.float32)

        encoded_text = ort.OrtValue.ortvalue_from_numpy(encoded_text)
        pace = ort.OrtValue.ortvalue_from_numpy(pace)
        pitch = ort.OrtValue.ortvalue_from_numpy(pitch)

        # ortvalue.device_name()  # 'cpu'
        # ortvalue.shape()        # shape of the numpy array X
        # ortvalue.data_type()    # 'tensor(float)'
        # ortvalue.is_tensor()    # 'True'
        # np.array_equal(ortvalue.numpy(), X)  # 'True'

        results = self.ort_session.run(None, input_feed= {"text": encoded_text, "pitch":pitch, "pace": pace})

        spec = results[0]

        return spec

    def infer_hifigan(self, spec: np.array):
        input_shapes = spec.shape

        B = input_shapes[0]
        inputs = []
        outputs = []
        bindings = []
        self.context.set_input_shape("spec", spec.shape)
        self.context.infer_shapes()

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
                temp_shape = self.context.get_tensor_shape(tensor_name)
                
                # temp_shape = (1,)  # Placeholder, adjust if necessary
                size = trt.volume(temp_shape)
                # print(temp_shape, size, dtype)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                outputs.append(HostDeviceMem(host_mem, device_mem))
                bindings.append(int(device_mem))

        # Transfer inputs to device
        for i in range(len(inputs)):
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, self.stream)

        # Set tensor address for each input/output
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])

        self.context.execute_async_v3(self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return outputs[0].host
