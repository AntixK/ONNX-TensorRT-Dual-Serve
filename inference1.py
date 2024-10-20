import re
import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm
from box import Box

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
# from torch.nn.functional import l1_loss
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator

from common import gpu_affinity
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
# from fastpitch.pitch_transform import pitch_transform_custom
# from hifigan.data_function import MAX_WAV_VALUE, mel_spectrogram
from hifigan.models import Denoiser


CHECKPOINT_SPECIFIC_ARGS = [
    'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
    'symbol_set', 'max_wav_value', 'prepend_space_to_text',
    'append_space_to_text']

def load_config(config_file:Path) -> Box:
    config = Box.from_yaml(filename=config_file)
    return config

    
def get_model(model_name, model_config, device, bn_uniform_init=False,
              forward_is_infer=False, jitable=False):
    """Chooses a model based on name"""
    del bn_uniform_init  # unused (old name: uniform_initialize_bn_weight)

    if model_name == 'FastPitch':
        if jitable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)

    elif model_name == 'HiFi-GAN':
        model = Generator(model_config)
    else:
        raise NotImplementedError(model_name)

    if forward_is_infer and hasattr(model, 'infer'):
        model.forward = model.infer

    return model.to(device)

def get_model_config(model_name: str) -> dict:
    CONFIG_FILE = Path("config.yaml")
    config = load_config(CONFIG_FILE)

    if model_name == 'FastPitch':
        return config.model_config.fastpitch.to_dict()
    elif model_name == 'HiFi-GAN':
        return config.model_config.hifigan.to_dict()
    else:
        raise NotImplementedError(model_name)

def load_model_from_ckpt(checkpoint_data, model, key:str='state_dict') -> tuple:

    if key is None:
        return checkpoint_data['model'], None

    sd = checkpoint_data[key]
    sd = {re.sub('^module\.', '', k): v for k, v in sd.items()}
    status = model.load_state_dict(sd, strict=False)
    return (model, status)


def load_and_setup_model(model_name:str, checkpoint, amp, device,
                         forward_is_infer=False, jitable=False):
    if checkpoint is not None:
        ckpt_data = torch.load(checkpoint)
        print(f'{model_name}: Loading {checkpoint}...')
        ckpt_config = ckpt_data.get('config')
        if ckpt_config is None:
            print(f'{model_name}: No model config in the checkpoint; using args.')
        else:
            print(f'{model_name}: Found model config saved in the checkpoint.')
    else:
        ckpt_config = None
        ckpt_data = {}

    model_config = get_model_config(model_name)

    model = get_model(model_name, model_config, device,
                      forward_is_infer=forward_is_infer,
                      jitable=jitable)

    if checkpoint is not None:
        key = 'generator' if model_name == 'HiFi-GAN' else 'state_dict'
        model, status = load_model_from_ckpt(ckpt_data, model, key)

    if model_name == 'HiFi-GAN':
        model.remove_weight_norm()

    if amp:
        model.half()

    model.eval()
    return model.to(device), model_config, ckpt_data.get('train_setup', {})


def load_and_setup_ts_model(model_name, checkpoint, amp, device=None):
    print(f'{model_name}: Loading TorchScript checkpoint {checkpoint}...')
    model = torch.jit.load(checkpoint).eval()
    if device is not None:
        model = model.to(device)
    
    if amp:
        model.half()
    elif next(model.parameters()).dtype == torch.float16:
        raise ValueError('Trying to load FP32 model,'
                         'TS checkpoint is in FP16 precision.')
    return model

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true',
                        help='Save generator outputs to disk')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--use-cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--l2-promote', action='store_true',
                        help='Increase max fetch granularity of GPU L2 cache')
    parser.add_argument('--fastpitch', type=str, default=None, required=False,
                        help='Full path to the spectrogram generator .pt file '
                             '(skip to synthesize from ground truth mels)')
    parser.add_argument('--hifigan', type=str, default=None, required=False,
                        help='Full path to a HiFi-GAN model .pt file')
    parser.add_argument('-d', '--denoising-strength', default=0.0, type=float,
                        help='Capture and subtract model bias to enhance audio')
    parser.add_argument('--hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--window-length', type=int, default=1024,
                        help='STFT win length for denoiser and mel loss')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        choices=[22050, 44100], help='Sampling rate')
    parser.add_argument('--max_wav_value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--use-amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--num-repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Run inference with TorchScript model (convert to TS if needed)')
    parser.add_argument('--checkpoint-format', type=str,
                        choices=['pyt', 'ts'], default='pyt',
                        help='Input checkpoint format (PyT or TorchScript)')
    parser.add_argument('--torch-tensorrt', action='store_true',
                        help='Run inference with Torch-TensorRT model (compile beforehand)')
    parser.add_argument('--report-mel-loss', action='store_true',
                        help='Report mel loss in metrics')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker-id', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--affinity', type=str, default='single',
                        choices=['socket', 'single', 'single_unique',
                                 'socket_unique_interleaved',
                                 'socket_unique_continuous',
                                 'disabled'],
                        help='type of CPU affinity')
    parser.add_argument('--pace', type=float, default=1.0,
                        help='Adjust the pace of speech')

    txt = parser.add_argument_group('Text processing parameters')
    txt.add_argument('--text-cleaners', type=str, nargs='*',
                     default=['english_cleaners_v2'],
                     help='Type of text cleaners for input text')
    txt.add_argument('--symbol-set', type=str, default='english_basic',
                     help='Define symbol set for input text')
    txt.add_argument('--p-arpabet', type=float, default=0.0, help='')
    txt.add_argument('--heteronyms-path', type=str,
                     default='cmudict/heteronyms', help='')
    txt.add_argument('--cmudict-path', type=str,
                     default='cmudict/cmudict-0.7b', help='')
    return parser

def get_args():
    config = load_config(Path("config.yaml"))

    return config.inference


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c: f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, 
                           device, 
                           symbol_set, 
                           text_cleaners,
                           batch_size=128, 
                           p_arpabet=0.0):
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


def main():
    """
    Launches text-to-speech inference on a single GPU.
    """
    # parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
    #                                  allow_abbrev=False)
    # parser = parse_args(parser)
    # args, unk_args = parser.parse_known_args()

    args = get_args()

    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        # print(nproc_per_node)
        affinity = gpu_affinity.set_affinity(
            0,
            nproc_per_node,
            args.affinity
        )
        print(f'Thread affinity: {affinity}')

    if args.l2_promote:
        l2_promote()
    torch.backends.cudnn.benchmark = args.use_cudnn_benchmark

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=False, exist_ok=True)

    log_fpath = str(Path(args.output_dir, 'nvlog_infer.json'))

    DLLogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
        JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
        StdOutBackend(Verbosity.VERBOSE, metric_format=stdout_metric_format)
    ])
    init_inference_metadata(args.batch_size)
    # [DLLogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    generator = None
    vocoder = None
    denoiser = None

    # is_ts_based_infer = args.torch_tensorrt or args.torchscript
    is_ts_based_infer = False

    # assert args.checkpoint_format == 'pyt' or is_ts_based_infer, \
    #     'TorchScript checkpoint can be used only for TS or Torch-TRT' \
    #     ' inference. Please set --torchscript or --torch-tensorrt flag.'

    
    def _load_pyt_or_ts_model(model_name, ckpt_path):

        # print(f"Checkpoint format: {model_name}, {args.checkpoint_format}")
        # if args.checkpoint_format == 'ts':
        #     model = load_and_setup_ts_model(model_name, ckpt_path,
        #                                            args.amp, device)
        #     model_train_setup = {}
        #     return model, model_train_setup
        model, _, model_train_setup = load_and_setup_model(
            model_name, ckpt_path, args.use_amp, device, forward_is_infer=True, jitable=is_ts_based_infer)

        # if is_ts_based_infer:
        #     model = torch.jit.script(model)
        return model, model_train_setup

    if args.fastpitch is not None:
        gen_name = 'fastpitch'
        generator, gen_train_setup = _load_pyt_or_ts_model('FastPitch',
                                                           args.fastpitch)

    if args.hifigan is not None:
        voc_name = 'hifigan'
        vocoder, voc_train_setup = _load_pyt_or_ts_model('HiFi-GAN',
                                                         args.hifigan)

        if args.denoising_strength > 0.0:
            denoiser = Denoiser(vocoder, win_length=args.window_length).to(device)

        # if args.torch_tensorrt:
        #     vocoder = convert_ts_to_trt('HiFi-GAN', vocoder, parser,
        #                                        args.amp, unk_args)

        def generate_audio(mel):
            audios = vocoder(mel).float()
            if denoiser is not None:
                audios = denoiser(audios.squeeze(1), args.denoising_strength)
            return audios.squeeze(1) * args.max_wav_value

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, args.heteronyms_path)

    gen_kw = {'pace': args.pace,
              'speaker': args.speaker_id,
              'pitch_tgt': None,
              'pitch_transform': None}


    # Prepare data
    fields = load_fields(args.input_file)
    batches = prepare_input_sequence(
        fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
        p_arpabet=args.p_arpabet)

    # Do warmup
    cycle = itertools.cycle(batches)
    # Use real data rather than synthetic - FastPitch predicts len
    for _ in tqdm(range(args.warmup_steps), 'Warmup'):
        with torch.no_grad():
            b = next(cycle)
            if generator is not None:
                mel, *_ = generator(b['text'])
            else:
                mel, mel_lens = b['mel'], b['mel_lens']
                if args.use_amp:
                    mel = mel.half()
            if vocoder is not None:
                audios = generate_audio(mel)

    gen_measures = MeasureTime(cuda=args.use_cuda)
    vocoder_measures = MeasureTime(cuda=args.use_cuda)

    all_utterances = 0
    all_samples = 0
    all_batches = 0
    all_letters = 0
    all_frames = 0
    gen_mel_loss_sum = 0
    voc_mel_loss_sum = 0

    reps = args.num_repeats
    log_enabled = reps == 1
    log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

    for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
        for b in batches:

            # Generate mel spectrograms from FastPitch
            with torch.no_grad(), gen_measures:
                mel, mel_lens, *_ = generator(b['text'], **gen_kw)

            gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
            all_letters += b['text_lens'].sum().item()
            all_frames += mel.size(0) * mel.size(2)
            log(rep, {f"{gen_name}_frames/s": gen_infer_perf})
            log(rep, {f"{gen_name}_latency": gen_measures[-1]})

            # Generate audio from mel spectrograms using Hifigan vocoder
            with torch.no_grad(), vocoder_measures:
                audios = generate_audio(mel)

            vocoder_infer_perf = (
                audios.size(0) * audios.size(1) / vocoder_measures[-1])

            log(rep, {f"{voc_name}_samples/s": vocoder_infer_perf})
            log(rep, {f"{voc_name}_latency": vocoder_measures[-1]})

            if args.output_dir is not None and reps == 1:
                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * args.hop_length]

                    if args.fade_out:
                        fade_len = args.fade_out * args.hop_length
                        fade_w = torch.linspace(1.0, 0.0, fade_len)
                        audio[-fade_len:] *= fade_w.to(audio.device)

                    audio = audio / torch.max(torch.abs(audio))
                    fname = b['output'][i] if 'output' in b else f'audio_{all_utterances + i}.wav'
                    audio_path = Path(args.output_dir, fname)
                    write(audio_path, args.sampling_rate, audio.cpu().numpy())

            # if generator is not None:
            log(rep, {"latency": (gen_measures[-1] + vocoder_measures[-1])})

            all_utterances += mel.size(0)
            all_samples += mel_lens.sum().item() * args.hop_length
            all_batches += 1

    log_enabled = True
    if generator is not None:
        gm = np.sort(np.asarray(gen_measures))
        rtf = all_samples / (all_utterances * gm.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * gm.mean() * args.sampling_rate)
        log((), {f"avg_{gen_name}_tokens/s": all_letters / gm.sum()})
        log((), {f"avg_{gen_name}_frames/s": all_frames / gm.sum()})
        log((), {f"avg_{gen_name}_latency": gm.mean()})
        log((), {f"avg_{gen_name}_RTF": rtf})
        log((), {f"avg_{gen_name}_RTF@{args.batch_size}": rtf_at})
        log((), {f"90%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
        log((), {f"95%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
        log((), {f"99%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})
        if args.report_mel_loss:
            log((), {f"avg_{gen_name}_mel-loss": gen_mel_loss_sum / all_utterances})
    if vocoder is not None:
        vm = np.sort(np.asarray(vocoder_measures))
        rtf = all_samples / (all_utterances * vm.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * vm.mean() * args.sampling_rate)
        log((), {f"avg_{voc_name}_samples/s": all_samples / vm.sum()})
        log((), {f"avg_{voc_name}_latency": vm.mean()})
        log((), {f"avg_{voc_name}_RTF": rtf})
        log((), {f"avg_{voc_name}_RTF@{args.batch_size}": rtf_at})
        log((), {f"90%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.90) / 2) * vm.std()})
        log((), {f"95%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.95) / 2) * vm.std()})
        log((), {f"99%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.99) / 2) * vm.std()})
        if args.report_mel_loss:
            log((), {f"avg_{voc_name}_mel-loss": voc_mel_loss_sum / all_utterances})
    if generator is not None and vocoder is not None:
        m = gm + vm
        rtf = all_samples / (all_utterances * m.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * m.mean() * args.sampling_rate)
        log((), {"avg_samples/s": all_samples / m.sum()})
        log((), {"avg_letters/s": all_letters / m.sum()})
        log((), {"avg_latency": m.mean()})
        log((), {"avg_RTF": rtf})
        log((), {f"avg_RTF@{args.batch_size}": rtf_at})
        log((), {"90%_latency": m.mean() + norm.ppf((1.0 + 0.90) / 2) * m.std()})
        log((), {"95%_latency": m.mean() + norm.ppf((1.0 + 0.95) / 2) * m.std()})
        log((), {"99%_latency": m.mean() + norm.ppf((1.0 + 0.99) / 2) * m.std()})
    DLLogger.flush()


if __name__ == '__main__':
    main()
