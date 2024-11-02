import re
import torch
from loguru import logger
from box import Box
import onnx
import onnxruntime as ort
import tensorrt as trt
from pathlib import Path
from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator
from typing import Callable, Union

def get_args(config_file: Union[Path, str]) -> Box:
    """
    Get configuration arguments

    Args:
        config_file (Union[Path, str]): Path to configuration file

    Returns:
        config (Box): Configuration arguments
    """
    return Box.from_yaml(filename=config_file)

def get_model(model_name:str,
              model_args: Box,
              args: Box,
              device: torch.device,
              jittable: bool =False) -> Callable:

    """
    Get model based on model name

    Args:
        model_name: Name of the model
        model_args: Model configuration
        args: Arguments
        device: Device to run model
        jittable: Whether to use JIT or not

    Returns:
        model: Model instance

    Raises:
        NotImplementedError: If model name is not supported
    """

    if model_name == 'FastPitch':
        model_config = model_args.fastpitch
        if jittable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)
        checkpoint_path = args.fastpitch

    elif model_name == 'HiFi-GAN':
        model_config = model_args.hifigan
        model = Generator(model_config)
        checkpoint_path = args.hifigan

    else:
        raise NotImplementedError(model_name)

    if hasattr(model, 'infer'):
        model.forward = model.infer

    ckpt_data = torch.load(checkpoint_path, weights_only=False)

    assert 'generator' in ckpt_data or 'state_dict' in ckpt_data

    if model_name == 'HiFi-GAN':
        sd = ckpt_data['generator']
    else:
        sd = ckpt_data['state_dict']

    # Remove 'module.' prefix in checkpoint state_dict
    sd = {re.sub('^module\.', '', k): v for k, v in sd.items()}
    status = model.load_state_dict(sd, strict=False)

    if model_name == 'HiFi-GAN':
        model.remove_weight_norm()

    if args.use_amp:
        model.half()

    model.eval()
    model = model.to(device)

    return model

def convert_fastpitch(model_args:Box, args: Box, device = "cuda"):
    # Initialie MelSpectrogram Generator (FastPitch)
    generator = get_model(model_name='FastPitch',
                            model_args=model_args,
                            args=args,
                            device=device,
                            jittable=True)

    logger.info(f'Loaded FastPitch model from {args.fastpitch}')

    # Convert FastPitch to ONNX
    # generator = generator
    generator.eval()
    text_input = torch.randint(3, 5, (1, 122), dtype=torch.int32).to(device)
    pace = torch.tensor(1.0, dtype=torch.float32).to(device)
    # Check if model can run
    with torch.no_grad():
        mel_output = generator(text_input)
        # print([m.shape for m in mel_output])
        # logger.info(f"Mel Output Shape: {mel_output.shape}")

    # print(text_input.shape)

    gen_onnx_model = torch.onnx.export(
                        generator,
                        (text_input, pace),
                        'pretrained_models/fastpitch.onnx',
                        export_params=True,
                        verify=True,
                        opset_version=20,
                        do_constant_folding=True,
                        input_names = ['text'],
                        output_names = ['mel'],
                        dynamic_axes={'text' : {0 : 'batch_size',
                                                1 : "seq_length",
                        },
                                       'mel' : {0 : "batch_size",
                                                2 : "mel_length",
                                       }},
    )

    # Sanity check
    onnx_model = onnx.load('pretrained_models/fastpitch.onnx')
    onnx.checker.check_model(onnx_model)
    del onnx_model

    logger.info("Converted FastPitch to ONNX Model")


def convert_hifigan(model_args:Box, args: Box, device = "cuda"):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

    vocoder = get_model(model_name='HiFi-GAN',
                             model_args=model_args,
                             args=args,
                             device=device,
                             jittable=True)

    vocoder = vocoder

    logger.info(f'Loaded HiFi-GAN model from {args.hifigan}')

    # Convert to ONNX
    mel_input = torch.randn(1, 80, 661).to(device)
    if args.use_amp:
        mel_input = mel_input.half()

    # Cannot use torch.dynamo to export to ONNX
    # as it does not support custom input and output naming
    # and it is a PITA to bind torch tensorts to ONNX
    # See this Github Issue:
    #  https://github.com/pytorch/pytorch/issues/107355
    # voc_onnx_model = torch.onnx.dynamo_export(vocoder, mel_input)


    onnxfile = 'pretrained_models/hifigan.onnx'
    voc_onnx_model = torch.onnx.export(
                        vocoder,
                        mel_input,
                        onnxfile,
                        export_params=True,
                        do_constant_folding=True,
                        input_names = ['mel'],
                        output_names = ['audio'],
                        dynamic_axes={'mel' : {0 : 'batch_size'},
                                    'audio' : {0 : 'batch_size'}})

    # Sanity check
    onnx_model = onnx.load(onnxfile)
    onnx.checker.check_model(onnx_model)
    del onnx_model

    major, minor, patch = trt.__version__.split('.')

    logger.info(f"TensorRT Version: {major}.{minor}.{patch}")
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #we have enabled the explicit Batch


    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as builder_config:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<< 30)

        if args.use_amp:
            logger.info("Using FP16 Precision")
            builder_config.set_flag(trt.BuilderFlag.FP16)

        logger.info("Parsing ONNX file.")
        with open(onnxfile, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                logger.error("Failed to parse ONNX file.")
                return None

        # Set dynamic shapes
        logger.info("Setting optimization profile.")
        profile = builder.create_optimization_profile()
        # TODO: Set proper dynamic shapes
        profile.set_shape("mel", (1, 80, 661), (4, 80, 661), (8, 80,661))
        builder_config.add_optimization_profile(profile)
        builder_config.default_device_type = trt.DeviceType.GPU

        logger.info("Building TensorRT engine. This may take a few minutes.")
        serialized_engine = builder.build_serialized_network(network, builder_config)

        with open('pretrained_models/hifigan.plan', 'wb') as f:
            f.write(serialized_engine)
            logger.info("TensorRT engine saved to pretrained_models/hifigan.plan")

        return serialized_engine

    # with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_creation_flag) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MiB
    #     # Parse model file
    #     print("Loading ONNX file from path {}...".format(onnxfile))
    #     with open(onnxfile, "rb") as model:
    #         print("Beginning ONNX file parsing")
    #         if not parser.parse(model.read()):
    #             print("ERROR: Failed to parse the ONNX file.")
    #             for error in range(parser.num_errors):
    #                 print(parser.get_error(error))
    #             return None
    #         # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
    #         network.get_input(0).shape = [80, 661]
    #         print("Completed parsing of ONNX file")
    #         print(
    #             "Building an engine from file {}; this may take a while...".format(
    #                 onnxfile
    #             )
    #         )
    #         plan = builder.build_serialized_network(network, config)
    #         engine = runtime.deserialize_cuda_engine(plan)
    #         print("Completed creating Engine")
    #         with open('pretrained_models/hifigan.plan', "wb") as f:
    #             f.write(plan)
    #         return engine

    # if os.path.exists(engine_file_path):
    #     # If a serialized engine exists, use it instead of building an engine.
    #     print("Reading engine from file {}".format(engine_file_path))
    #     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         return runtime.deserialize_cuda_engine(f.read())
    # else:
    #     return build_engine()


if __name__ == "__main__":
    config = get_args(Path("config.yaml"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args: Box = config.inference
    model_args: Box = config.model_config

    convert_fastpitch(model_args, args, device)
    convert_hifigan(model_args, args, device)
