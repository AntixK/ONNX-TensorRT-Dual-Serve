#!/usr/bin/env bash

wget https://raw.githubusercontent.com/NVIDIA/NeMo/263a30be71e859cee330e5925332009da3e5efbc/scripts/tts_dataset_files/heteronyms-052722 -O models/heteronyms
wget https://raw.githubusercontent.com/NVIDIA/NeMo/263a30be71e859cee330e5925332009da3e5efbc/scripts/tts_dataset_files/cmudict-0.7b_nv22.08 -O models/cmudict-0.7b
wget https://api.ngc.nvidia.com/v2/models/nvidia/fastpitch_pyt_fp32_ckpt_v1_1/versions/21.05.0/zip -O models/nvidia_fastpitch.zip
wget https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/zip -O models/nvidia_hifigan.zip

unzip models/nvidia_fastpitch.zip -d models/nvidia_fastpitch
unzip models/nvidia_hifigan.zip -d models/nvidia_hifigan