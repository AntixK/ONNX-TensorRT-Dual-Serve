import re
import itertools
import time
import sys
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from box import Box
import base64
import torch
import numpy as np
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

from loguru import logger

from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator

from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
warnings.filterwarnings("ignore")

from typing import List, Tuple, Callable, Union

import litserve as ls


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


def _convert_ts_to_trt_hifigan(ts_model: Callable,
                               use_amp: bool,
                               trt_min_opt_max_batch: Tuple,
                               trt_min_opt_max_hifigan_length: Tuple,
                               num_mels:int=80):
    import torch_tensorrt
    trt_dtype = torch.half if use_amp else torch.float
    print(f'Torch TensorRT: compiling HiFi-GAN for dtype {trt_dtype}.')
    min_shp, opt_shp, max_shp = zip(trt_min_opt_max_batch,
                                    (num_mels,) * 3,
                                    trt_min_opt_max_hifigan_length)
    compile_settings = {
        "inputs": [torch_tensorrt.Input(
            min_shape=min_shp,
            opt_shape=opt_shp,
            max_shape=max_shp,
            dtype=trt_dtype,
        )],
        "enabled_precisions": {trt_dtype},
        "require_full_compilation": True,
    }
    trt_model = torch_tensorrt.compile(ts_model, **compile_settings)
    print('Torch TensorRT: compilation successful.')
    return trt_model


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


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.time() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=self.cuda)

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
        self.generator = None
        self.vocoder = None
        self.is_ts_based_infer = True

        if self.args.l2_promote:
            l2_promote()

        cmudict.initialize(self.args.cmudict_path, self.args.heteronyms_path)

        # Initialie MelSpectrogram Generator (FastPitch)
        self.generator = get_model(model_name='FastPitch',
                                   model_args=self.model_args,
                                   args=self.args,
                                   device=self.device,
                                   jittable=self.is_ts_based_infer)

        self.gen_kw = {'pace': self.args.pace,
                       'speaker': self.args.speaker_id,
                       'pitch_tgt': None,}

        # Initialize Vocoder (HiFi-GAN)
        self.vocoder = get_model(model_name='HiFi-GAN',
                                 model_args=self.model_args,
                                 args=self.args,
                                 device=self.device,
                                 jittable=self.is_ts_based_infer)

        # Initialize Text Processor
        self.tp = get_text_processing(self.args.symbol_set,
                                         self.args.text_cleaners,
                                         self.args.p_arpabet)

        # Initilize measurement trackers
        self.gen_measures = MeasureTime(cuda=self.args.use_cuda)
        self.voc_measures = MeasureTime(cuda=self.args.use_cuda)



        # logger.remove(0)
        # log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <b>{message}</b>"
        # # logger.add(sys.stdout, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)

        # logger.add(f"{self.args.log_dir}/time.log", level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess the input text by encoding it into tensor.

        Args:
            text (List[str]): List of input text.

        Returns:
            Tuple: Tuple of encoded text tensor and text lengths tensor.
        """
        encoded_text = [torch.LongTensor(self.tp.encode_text(text))]
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

    def _convert_to_onnx(self):
        pass

    def _convert_to_trt(self):
        pass

    @property
    def generator_name(self)->str:
        return 'FastPitch'

    @property
    def vocoder_name(self)->str:
        return 'HiFi-GAN'

    @torch.inference_mode()
    def do_warmup(self):
        num_warmup_steps = self.args.warmup_steps
        batch_iter = itertools.cycle(self._get_warmup_batches())

        for _ in tqdm(range(num_warmup_steps), desc='Warmup', dynamic_ncols=True):
            b = next(batch_iter)
            mel, *_ = self.generator(b['text'], **self.gen_kw)
            # print(b['text'].size(), mel.size())
            audios = self._generate_audio_from_mel(mel)

    @torch.inference_mode()
    def run(self, encoded_text: torch.Tensor) -> Tuple:
        encoded_text = encoded_text.to(self.device)

        # with self.gen_measures:
        mel, mel_lens, *_ = self.generator(encoded_text, **self.gen_kw)

        # with self.voc_measures:
        audios = self.vocoder(mel).float()
        audios = audios.squeeze(1) * self.args.max_wav_value

        # logger.info(f"Generator-Inference-Time:{self.gen_measures[-1]:4.4f}s | Vocoder-Inference-Time:{self.voc_measures[-1]:4.4f}s")
        return (audios, mel, mel_lens)

    @torch.inference_mode()
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
        with open("logs/tts_server.log", "a+") as f:
            f.write(f"{key}:{value:.4f}\n")

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
        lit_api.log("Inference_time", elapsed)

        lit_api.log("Samples", lit_api.num_samples)
        lit_api.log("Utterances", lit_api.num_utterances)

        torch.cuda.memory._dump_snapshot(f"{lit_api.log_dir}/gpu_mem_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    # def (self):


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

        # self.pre_measures = MeasureTime(cuda=config.inference.use_cuda)
        # self.post_measures = MeasureTime(cuda=config.inference.use_cuda)

    def decode_request(self, request: dict) -> torch.Tensor:
        """
        Decode the JSON request from the client into
        text to be used by the Text2Speech model.
        """
        # with self.pre_measures:
        text = request['text']
        encoded_text = self.tts.preprocess_text(text)

        return encoded_text

    # def batch(self, decoded_requests: torch.Tensor) -> List[str]:
    #     return decoded_requests

    # def unbatch(self, responses):
    #     pass

    def predict(self, encoded_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio, mel, mel_lens = self.tts.run(encoded_text)

        # Update throughput tracker
        self.num_samples = mel_lens.sum().item() * self.tts.args.hop_length
        self.num_utterances = mel.size(0)

        # print(encoded_text.size(), mel.size())
        return (audio, mel, mel_lens)

    def encode_response(self, model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict:

        # with self.post_measures:
        audio, mel, mel_lens = model_output

        audio = self.tts.postprocess(audio, mel_lens)
        # Convert audio tensor into a serialized base64 string
        audio = base64.b64encode(audio.cpu().numpy()).decode("utf-8")

        return {"content": audio,
                "content_type": "audio/wav",
                "content_encoding": "utf-8",
                "dtype": "float32"}


if __name__ == '__main__':
    # # main()

    # torch.cuda.empty_cache()

    # config = get_args(Path("config.yaml"))

    # tts = Text2Speech(config)
    # print(tts)

    # tts.do_warmup()
    # tts("Yo! waddup mayne? How you doing?")


    ttsapi = TTSServer()
    # server = ls.LitServer(ttsapi, max_batch_size=1, batch_timeout=0.01)
    server = ls.LitServer(ttsapi,
                          accelerator="gpu",
                          callbacks=[InferenceTimeLogger()],
                          loggers=FileLogger())

    server.run(port=7008)
