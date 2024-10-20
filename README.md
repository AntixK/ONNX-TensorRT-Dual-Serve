# ONNX-TensorRT-Dual-Serve

## Download Models (Trained on LJSpeech Dataset)
```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/fastpitch_pyt_fp32_ckpt_v1_1/versions/21.05.0/zip -O models/nvidia_fastpitch_210824.pt

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/zip -O models/hifigan_gen_checkpoint_10000_ft.pt
```

- **FastPitch** is NVIDIA's fully-parallel (non-autoregressive) transformer architecture.

## Convert Models to ONNX and TensorRT

## Create a Triton Server


```
import soundfile as sf
parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
audio = model.convert_spectrogram_to_audio(spec=spectrogram)
```

E2E system:

test input -> generate spectrogram (FastPitch) -> Generate audio (HiFiGAN)


references:
1. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_e2e_fastpitchhifigan
2. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_hifigan
3. https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1150/user-guide/docs/install.html#:~:text=The%20Triton%20Inference%20Server%20is,Docker%20and%20nvidia%2Ddocker%20installed.
4. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/export.html
5. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/checkpoints.html
6. https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
7. https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/inference.py
8. https://www.youtube.com/embed/YHloC_py1QM
