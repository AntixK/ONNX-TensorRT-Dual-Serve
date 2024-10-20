import re
import itertools
import time
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from box import Box

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence


from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator

from common import gpu_affinity
from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
import warnings
warnings.filterwarnings("ignore")

from typing import List, Tuple, Callable, Union


def load_config(config_file:Path) -> Box:
    config = Box.from_yaml(filename=config_file)
    return config

def get_model(model_name:str, 
              model_args: Box, 
              args: Box, 
              device: torch.device,
              jittable: bool =False) -> Callable:

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

def get_args():
    config = load_config(Path("config.yaml"))
    return config


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
        
    def preprocess_text(self, text: List[str]):
        tp = get_text_processing(self.args.symbol_set, self.args.text_cleaners, self.args.p_arpabet)
        encoded_text = [torch.LongTensor(tp.encode_text(t)) for t in text]
        order = np.argsort([-t.size(0) for t in encoded_text])
        encoded_text = [encoded_text[i] for i in order]
        text_lens = torch.LongTensor([t.size(0) for t in encoded_text])
        encoded_text = pad_sequence(encoded_text, batch_first=True)

        return encoded_text, text_lens

    
    def postprocess(self, audios: torch.Tensor, mel_lens: torch.Tensor) -> List[torch.Tensor]:
        res_audios = []
        for i, audio in enumerate(audios):
            audio = audio[:mel_lens[i].item() * self.args.hop_length]

            if self.args.fade_out > 0:
                fade_len = self.args.fade_out * self.args.hop_length
                fade_w = torch.linspace(1.0, 0.0, fade_len)
                res_audio = audio.clone()
                res_audio[-fade_len:] = audio[-fade_len:] * fade_w.to(audio.device)

            audio = audio / torch.max(torch.abs(audio))

            res_audios.append(audio)
        return res_audios
    
    def save_audio(self, audios: List[torch.Tensor]):

        Path(self.args.output_dir).mkdir(parents=False, exist_ok=True)
        
        for i, audio in enumerate(audios):
            fname = f'audio_{i}.wav'
            audio_path = Path(self.args.output_dir, fname)
            write(audio_path, self.args.sampling_rate, audio.cpu().numpy())

    def _get_warmup_batches(self):
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
            audios = self._generate_audio_from_mel(mel)

    @torch.inference_mode()
    def __call__(self, text:str) -> List[torch.Tensor]:
        # Split text into multiple sentences
        if len(text) > 1000:
            text = split_into_sentences(text)
        else:
            text = [text]

        encoded_text, _ = self.preprocess_text(text)
        encoded_text = encoded_text.to(self.device)
        mel, mel_lens, *_ = self.generator(encoded_text, **self.gen_kw)
        audios = self._generate_audio_from_mel(mel)

        res_audio = self.postprocess(audios, mel_lens)

        self.save_audio(res_audio)

        return res_audio

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}\n(Text -> {self.generator_name} -> {self.vocoder_name} -> Audio)'


# def main():
#     torch.cuda.empty_cache()

#     config = get_args()

#     args = config.inference
#     model_args = config.model_config


#     assert args.hifigan is not None or args.fastpitch is not None, \
#         'Both FastPitch or HiFi-GAN models must be provided.'

#     if args.affinity != 'disabled':
#         nproc_per_node = torch.cuda.device_count()
#         affinity = gpu_affinity.set_affinity(
#             0,
#             nproc_per_node,
#             args.affinity
#         )
#         # print(f'Thread affinity: {affinity}')

#     if args.l2_promote:
#         l2_promote()

#     torch.backends.cudnn.benchmark = args.use_cudnn_benchmark

#     if args.output_dir is not None:
#         Path(args.output_dir).mkdir(parents=False, exist_ok=True)

#     log_fpath = str(Path(args.output_dir, 'nvlog_infer.json'))

#     device = torch.device('cuda' if args.use_cuda else 'cpu')

#     generator = None
#     vocoder = None

#     is_ts_based_infer = True

#     # Get MelSpectrogram Generator (FastPitch)
#     gen_name = 'fastpitch'
#     generator = get_model(model_name='FastPitch',
#                             model_args= model_args,
#                             args=args,
#                             device=device,
#                             jittable=is_ts_based_infer)

#     # Get Vocoder (HiFi-GAN)
#     voc_name = 'hifigan'
#     vocoder = get_model(model_name='HiFi-GAN',
#                         model_args= model_args,
#                         args=args,
#                         device=device,
#                         jittable=is_ts_based_infer)
    
#     # vocoder = _convert_ts_to_trt_hifigan(vocoder, 
#     #                                      args.use_amp, 
#     #                                      trt_min_opt_max_batch=(1, 8, 16),
#     #                                      trt_min_opt_max_hifigan_length=(1, 80, 8192))

#     def generate_audio(mel):
#         audios = vocoder(mel).float()
#         return audios.squeeze(1) * args.max_wav_value

#     if args.p_arpabet > 0.0:
#         cmudict.initialize(args.cmudict_path, args.heteronyms_path)

#     gen_kw = {'pace': args.pace,
#               'speaker': args.speaker_id,
#               'pitch_tgt': None,
#               }


#     # Prepare data
#     fields = load_fields(args.input_file)
#     batches = prepare_input_sequence(
#         fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
#         p_arpabet=args.p_arpabet)

#     # Do warmup
#     cycle = itertools.cycle(batches)
#     # Use real data rather than synthetic - FastPitch predicts len
#     for _ in tqdm(range(args.warmup_steps), 'Warmup'):
#         with torch.no_grad():
#             b = next(cycle)
#             mel, *_ = generator(b['text'])
#             audios = generate_audio(mel)

#     gen_measures = MeasureTime(cuda=args.use_cuda)
#     vocoder_measures = MeasureTime(cuda=args.use_cuda)

#     all_utterances = 0
#     all_samples = 0
#     all_batches = 0
#     all_letters = 0
#     all_frames = 0

#     reps = args.num_repeats
   
#     print(f'Inference: {reps} repetitions')
#     for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
#         for b in batches:

#             # Generate mel spectrograms from FastPitch
#             with torch.no_grad(), gen_measures:
#                 mel, mel_lens, *_ = generator(b['text'], **gen_kw)

#             gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
#             all_letters += b['text_lens'].sum().item()
#             all_frames += mel.size(0) * mel.size(2)
          

#             # Generate audio from mel spectrograms using Hifigan vocoder
#             with torch.no_grad(), vocoder_measures:
#                 audios = generate_audio(mel)

#             vocoder_infer_perf = (
#                 audios.size(0) * audios.size(1) / vocoder_measures[-1])


#             if args.output_dir is not None and rep == reps-1:
#                 for i, audio in enumerate(audios):
#                     audio = audio[:mel_lens[i].item() * args.hop_length]

#                     if args.fade_out:
#                         fade_len = args.fade_out * args.hop_length
#                         fade_w = torch.linspace(1.0, 0.0, fade_len)
#                         audio[-fade_len:] *= fade_w.to(audio.device)

#                     audio = audio / torch.max(torch.abs(audio))
#                     fname = b['output'][i] if 'output' in b else f'audio_{all_utterances + i}.wav'
#                     audio_path = Path(args.output_dir, fname)
#                     write(audio_path, args.sampling_rate, audio.cpu().numpy())


#             all_utterances += mel.size(0)
#             all_samples += mel_lens.sum().item() * args.hop_length
#             all_batches += 1


if __name__ == '__main__':
    # main()

    torch.cuda.empty_cache()

    config = get_args()

    tts = Text2Speech(config)
    print(tts)

    tts.do_warmup()

    # tts.preprocess_text(["Yo! waddup mayne? How you doing?"])
    tts("Yo! waddup mayne? How you doing?")

