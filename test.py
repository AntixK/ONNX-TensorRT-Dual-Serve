# from fastpitch.model import FastPitch
# from hifigan.models import Generator
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf
from scipy.io.wavfile import write

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


MODEL_DIR = Path("models")
torch.hub.set_dir(MODEL_DIR)
fastpitch, generator_train_setup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch')

hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')

fastpitch.to(device)
hifigan.to(device)
denoiser.to(device)


tp = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_textprocessing_utils', cmudict_path="cmudict-0.7b", heteronyms_path="heteronyms")



text = "Say this smoothly, to prove you are not a robot."

batches = tp.prepare_input_sequence([text], batch_size=1)

gen_kw = {'pace': 1.0,
          'speaker': 0,
          'pitch_tgt': None,
          'pitch_transform': None}
denoising_strength = 0.005


for batch in batches:
    with torch.no_grad():
        mel, mel_lens, *_ = fastpitch(batch['text'].to(device), **gen_kw)
        audios = hifigan(mel).float()
        audios = denoiser(audios.squeeze(1), denoising_strength)
        audios = audios.squeeze(1) * vocoder_train_setup['max_wav_value']

audio_numpy = audios[0].cpu().numpy()

# sf.write("speech.wav", audio_numpy, 22050)
write("audio.wav", vocoder_train_setup['sampling_rate'], audio_numpy)