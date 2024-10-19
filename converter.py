from fastpitch.model import FastPitch
from hifigan.models import Generator

import sys

# sys.path.append("NVIDIA_DeepLearningExamples_torchhub/PyTorch/SpeechSynthesis/FastPitch/common")
# from models.NVIDIA_DeepLearningExamples_torchhub.PyTorch.SpeechSynthesis.FastPitch.hifigan.models import Generator

import re
import torch
from pathlib import Path
from box import Box

import warnings
warnings.filterwarnings("ignore")



CONFIG_FILE = Path("config.yaml")
CHECKPOINT_DIR = Path("models")

def load_config(config_file:Path) -> Box:
    config = Box.from_yaml(filename=config_file)
    return config

def convert_to_onnx(model, checkpoint_path:Path):
    pass

def get_statedict(checkpoint_path:Path, key:str="state_dict") -> dict:
    checkpoint = torch.load(checkpoint_path)

    # print(checkpoint['generator'].keys())
    statedict = checkpoint[key]
    statedict = {re.sub('^module\.', '', k): v for k, v in statedict.items()}

    return statedict

def load_model_from_ckpt(checkpoint_data, model, key='state_dict'):

    if key is None:
        return checkpoint_data['model'], None

    sd = checkpoint_data[key]
    sd = {re.sub('^module\.', '', k): v for k, v in sd.items()}
    status = model.load_state_dict(sd, strict=False)
    return model, status

config = load_config(CONFIG_FILE)

# print(config)

f = FastPitch(**config.model_config.fastpitch)
h = Generator(config.model_config.hifigan)
if hasattr(h, 'infer'):
    h.forward = h.infer


# Load the checkpoints
# print(CHECKPOINT_DIR / config.checkpoint_config.fastpitch)
# fastpitch_weights = get_statedict(CHECKPOINT_DIR / config.checkpoint_config.fastpitch)
# f.load_state_dict(fastpitch_weights, strict=False)


# hifigan_weights = torch.load("pretrained_models/hifigan/hifigan_gen_checkpoint_6500.pt")["generator"]

# Extract Conv weights for the given version

conv_version = config.model_config.hifigan.resblock

# statedict = {}
# for k,v in hifigan_weights.items():
#     if f"convs{conv_version}" in k:
#         print(k)
#         k = k.replace("convs1", "convs")
#         print(k)

#     statedict[k] = v


# statedict = {re.sub('^module\.', '', k): v for k, v in statedict.items()}


# print(hifigan_weights.keys())

# print(h.state_dict().keys())
# status = h.load_state_dict(hifigan_weights, strict=False)

# h.remove_weight_norm()
ckpt_data = torch.load("pretrained_models/hifigan/hifigan_gen_checkpoint_10000_ft.pt")
key = 'generator' 
model, status = load_model_from_ckpt(ckpt_data, h, key)
