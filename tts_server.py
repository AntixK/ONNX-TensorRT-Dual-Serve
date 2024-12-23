
import torch
import torch_tensorrt
import tensorrt as trt
import re
import itertools
import time
import sys
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from box import Box
import base64
import pickle


import numpy as np
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
from loguru import logger

from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator

from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote

warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import List, Tuple, Callable, Union, Dict, Any

import onnx
import onnxruntime as ort

import litserve as ls

warnings.filterwarnings("ignore")


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

def get_args(config_file: Union[Path, str]) -> Box:
    """
    Get configuration arguments

    Args:
        config_file (Union[Path, str]): Path to configuration file

    Returns:
        config (Box): Configuration arguments
    """
    return Box.from_yaml(filename=config_file)


def load_fields(fpath) -> dict:
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c: f for c, f in zip(columns, fields)}

def prepare_input_sequence(fields: dict,
                           device: torch.device,
                           symbol_set: str,
                           text_cleaners: List[str],
                           batch_size:int =128,
                           p_arpabet: float =0.0):
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)
    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def split_into_sentences(paragraph: str) -> List[str]:
    # Regular expression pattern
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, paragraph)
    return sentences


class Text2Speech:
    def __init__(self, config: Box):
        self.args: Box = config.inference
        self.model_args: Box = config.model_config

        self.device = torch.device('cuda' if config.inference.use_cuda else 'cpu')

        logger.info(f'Using device: {self.device}')
        self.generator = None
        self.vocoder = None
        self.is_ts_based_infer = True

        if self.args.l2_promote:
            logger.info('L2 promotion is enabled')
            l2_promote()

        cmudict.initialize(self.args.cmudict_path, self.args.heteronyms_path)

        if self.args.use_amp:
            logger.info('AMP is enabled')

        # Initialie MelSpectrogram Generator (FastPitch)
        generator = get_model(model_name='FastPitch',
                                   model_args=self.model_args,
                                   args=self.args,
                                   device=self.device,
                                   jittable=self.is_ts_based_infer)

        logger.info(f'Loaded FastPitch model from {self.args.fastpitch}')

        # Convert FastPitch to ONNX
        generator = generator
        text_input = torch.randint(3, 5, (1, 122)).to(self.device)

        print(text_input.shape)


        gen_onnx_model = torch.onnx.export(
                            generator,
                            text_input,
                            'pretrained_models/fastpitch.onnx',
                            export_params=True,
                            do_constant_folding=True,
                            input_names = ['text'],
                            output_names = ['mel'],
                            dynamic_axes={'text' : {0 : 'batch_size'},
                                        'mel' : {0 : 'batch_size'}})

        generator = generator.to(self.device)
        ts_gen_model = torch.jit.script(generator)
        print(torch._dynamo.list_backends())
        torch._dynamo.reset()
        backend_kwargs = {
            "enabled_precisions": {torch.half if self.args.use_amp else torch.float},
            "cache_built_engines": True,
            "reuse_cached_engines": True,
            "debug": True,
            "min_block_size": 2,
            "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
            "optimization_level": 4,
            "use_python_runtime": False,
        }

        sample_inputs = torch.randint(3, 5, (1, 122)).to(self.device)
        trt_fastpitch = torch.compile(ts_gen_model,
            backend = "torch_tensorrt",
            options=backend_kwargs,
            dynamic=False)
        trt_fastpitch(sample_inputs)
        torch.jit.save(trt_fastpitch, 'pretrained_models/fastpitch_trt.plan')

        self.generator  = torch.jit.load('pretrained_models/fastpitch_trt.plan')
        # self.generator = trt_fastpitch


        logger.info("Converted FastPitch to Torch-TensorRT Model")

        self.gen_kw = {'pace': self.args.pace,
                       'speaker': self.args.speaker_id,
                       'pitch_tgt': None,}

        # Initialize Vocoder (HiFi-GAN)
        vocoder = get_model(model_name='HiFi-GAN',
                                 model_args=self.model_args,
                                 args=self.args,
                                 device=self.device,
                                 jittable=self.is_ts_based_infer)

        self.vocoder = vocoder

        logger.info(f'Loaded HiFi-GAN model from {self.args.hifigan}')

        if not Path('pretrained_models/hifigan.onnx').exists():

            # Convert to ONNX
            mel_input = torch.randn(1, 80, 661).to(self.device)
            if self.args.use_amp:
                mel_input = mel_input.half()

            # Cannot use torch.dynamo to export to ONNX
            # as it does not support custom input and output naming
            # and it is a PITA to bind torch tensorts to ONNX
            # See this Github Issue:
            #  https://github.com/pytorch/pytorch/issues/107355
            # voc_onnx_model = torch.onnx.dynamo_export(vocoder, mel_input)

            voc_onnx_model = torch.onnx.export(
                                vocoder,
                                mel_input,
                                'pretrained_models/hifigan.onnx',
                                export_params=True,
                                do_constant_folding=True,
                                input_names = ['mel'],
                                output_names = ['audio'],
                                dynamic_axes={'mel' : {0 : 'batch_size'},
                                            'audio' : {0 : 'batch_size'}})

        else:
            logger.info('ONNX model already exists, skipping conversion')

        # Sanity check
        onnx_model = onnx.load('pretrained_models/hifigan.onnx')
        onnx.checker.check_model(onnx_model)
        del onnx_model

        options = ort.SessionOptions()
        options.enable_profiling=False

        self.ort_session = ort.InferenceSession("pretrained_models/hifigan.onnx",
            sess_options=options,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

        logger.info("Converted HiFi-GAN to ONNX Model")

        # Initialize Text Processor
        self.tp = get_text_processing(self.args.symbol_set,
                                         self.args.text_cleaners,
                                         self.args.p_arpabet)

        logger.info(f'Loaded Text Processor with symbol set: {self.args.symbol_set}')

    def preprocess_text(self, text: Union[str, List]) -> torch.Tensor:
        """
        Preprocess the input text by encoding it into tensor.

        Args:
            text (List[str]): List of input text.

        Returns:
            Tuple: Tuple of encoded text tensor and text lengths tensor.
        """

        if isinstance(text, str):
            text = [text]

        encoded_text = [torch.LongTensor(self.tp.encode_text(t)) for t in text]
        encoded_text = pad_sequence(encoded_text, batch_first=True)
        return encoded_text

    def postprocess(self,
                    audio: torch.Tensor,
                    mel_lens: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the generated audio by applying fade out and normalization.

        Args:
            audio (torch.Tensor): The input audio tensor.

        Returns:
            List[torch.Tensor]: List of processed audio tensors.
        """

        audio = audio[:mel_lens.item() * self.args.hop_length]

        if self.args.fade_out > 0:
            fade_len = self.args.fade_out * self.args.hop_length

            assert fade_len <= audio.size(-1), f'Fade out length {fade_len} is longer than audio length {audio.size(-1)}.'

            fade_w = torch.linspace(1.0, 0.0, fade_len)
            res_audio = audio.clone()
            res_audio[:, -fade_len:] = audio[:, -fade_len:] * fade_w.to(audio.device)

        audio = audio / torch.max(torch.abs(audio))
        return audio  # Shape [1 x N]

    def save_audio(self, audios: List[torch.Tensor]):
        """
        Save the audio tensors to WAV files.

        Args:
            audios (List[torch.Tensor]): List of audio tensors to save.
        """

        Path(self.args.output_dir).mkdir(parents=False, exist_ok=True)

        for i, audio in enumerate(audios):
            fname = f'audio_{i}.wav'
            audio_path = Path(self.args.output_dir, fname)
            write(audio_path, self.args.sampling_rate, audio.cpu().numpy())

    def _get_warmup_batches(self):
        """
        Prepare warmup batches for the model.

        Returns:
            List: List of warmup batches.
        """
        # Prepare data
        fields = load_fields(self.args.input_file)
        batches = prepare_input_sequence(
            fields,
            self.device,
            self.args.symbol_set,
            self.args.text_cleaners,
            self.args.batch_size,
            p_arpabet=self.args.p_arpabet)

        return batches

    def _generate_audio_from_mel(self, mel) -> torch.Tensor:
        """
        Generate audio from mel spectrogram.

        Args:
            mel (torch.Tensor): The mel spectrogram tensor.

        Returns:
            torch.Tensor: The generated audio tensor.
        """
        audios = self.vocoder(mel).float()
        return audios.squeeze(1) * self.args.max_wav_value

    @property
    def generator_name(self)->str:
        return 'FastPitch'

    @property
    def vocoder_name(self)->str:
        return 'HiFi-GAN'

    @torch.no_grad()
    def do_warmup(self):
        num_warmup_steps = self.args.warmup_steps
        batch_iter = itertools.cycle(self._get_warmup_batches())

        for _ in tqdm(range(num_warmup_steps), desc='Warmup', dynamic_ncols=True):
            b = next(batch_iter)
            mel, *_ = self.generator(b['text'], **self.gen_kw)
            audios = self._generate_audio_from_mel(mel)


    # @torch.inference_mode()
    @torch.no_grad()
    def run(self, encoded_text: torch.Tensor) -> Tuple:
        encoded_text = encoded_text.to(self.device)

        mel, mel_lens, *_ = self.generator(encoded_text, **self.gen_kw)
        mel = mel.contiguous()

        # # Reference:
        # #  https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
        io_binding = self.ort_session.io_binding()

        io_binding.bind_input(
            name='mel',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(mel.shape),
            buffer_ptr=mel.data_ptr(),)

        out_shape = (1, 1, 169216)
        audio_tensor = torch.empty(out_shape,
                                   dtype=torch.float32,
                                   device=self.device).contiguous()
        io_binding.bind_output(
            name='audio',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(audio_tensor.shape),
            buffer_ptr=audio_tensor.data_ptr(),
        )

        # audio_tensor = self.vocoder(mel)

        audio_tensor = audio_tensor.float().squeeze(1) * self.args.max_wav_value
        return (audio_tensor, mel, mel_lens)

    # @torch.inference_mode()
    @torch.no_grad()
    def __call__(self, text:str) -> List[torch.Tensor]:
        encoded_text = self.preprocess_text(text)
        audios, mel, mel_lens = self.run(encoded_text)
        res_audio = self.postprocess(audios, mel_lens)

        self.save_audio(res_audio)

        return res_audio

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}\n(Text -> {self.generator_name} -> {self.vocoder_name} -> Audio)'


class FileLogger(ls.Logger):
    def process(self, key, value):
        # decode value dict
        metrics:dict = pickle.loads(base64.b64decode(value))
        metrics_str = ','.join([f"{k}:{v:.4f}" for k, v in metrics.items()])
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")

        with open("logs/tts_server.log", "a+") as f:
            # f.write(f"{time_now}|{key}:{value:.4f}\n")
            f.write(f"{time_now},{metrics_str}\n")

class InferenceTimeLogger(ls.Callback):
    def on_before_predict(self, lit_api):
        lit_api.num_samples =0
        lit_api.num_utterances = 0

        torch.cuda.memory._record_memory_history(max_entries=1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time

        metrics = {
            "Inference_time": elapsed,
            "Samples": lit_api.num_samples,
            "Utterances": lit_api.num_utterances
        }

        # Encode the metrics dict (workaround for LitServe API)
        metrics = pickle.dumps(metrics)
        encoded_metrics = base64.b64encode(metrics).decode('utf-8')

        lit_api.log("metrics", encoded_metrics)
        torch.cuda.memory._dump_snapshot(f"{lit_api.log_dir}/gpu_mem_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

class TTSServer(ls.LitAPI):
    def setup(self, device):
        config = get_args(Path("config.yaml"))

        self.log_dir = Path(config.inference.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tts = Text2Speech(config)
        self.tts.do_warmup()

        # Setup latency metric
        self.num_samples = 0
        self.num_utterances = 0

    def decode_request(self, request: dict) -> str:
        """
        Decode the JSON request from the client into
        text to be used by the Text2Speech model.
        """
        text = request['text']
        return text

    # def batch(self, decoded_requests: torch.Tensor) -> List[str]:
    #     # print(decoded_requests)
    #     return decoded_requests[0]

    # def unbatch(self, responses):
    #     pass

    def predict(self, text):
        encoded_text = self.tts.preprocess_text(text)

        audio, mel, mel_lens = self.tts.run(encoded_text)

        # Update throughput tracker
        self.num_samples = mel_lens.sum().item() * self.tts.args.hop_length
        self.num_utterances = mel.size(0)

        audio = self.tts.postprocess(audio, mel_lens)
        return audio.cpu().numpy()

    def encode_response(self, audio) -> dict:

        # Convert audio tensor into a serialized base64 string
        audio = base64.b64encode(audio).decode("utf-8")

        return {"content": audio,
                "content_type": "audio/wav",
                "content_encoding": "utf-8",
                "dtype": "float32"}


if __name__ == '__main__':
    # # main()

    torch.cuda.empty_cache()

    config = get_args(Path("config.yaml"))

    tts = Text2Speech(config)
    print(tts)

    tts.do_warmup()
    # tts("Yo! waddup mayne? How you doing?")
    #
    benchmark_data = np.genfromtxt("phrases/benchmark_8_128.tsv",
                                   delimiter="\t",
                                   dtype=str,
                                   skip_header=1)[:, -1]

    for i, text in enumerate(benchmark_data):
        tts(text)


    # ttsapi = TTSServer()
    # server = ls.LitServer(ttsapi,
    #                       max_batch_size=1,
    #                       batch_timeout=1.0,
    #                       accelerator="gpu",
    #                       callbacks=[InferenceTimeLogger()],
    #                       loggers=FileLogger())

    # server.run(port=7008)
