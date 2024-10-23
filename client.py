import requests
import base64
import numpy as np
from scipy.io.wavfile import write
from loguru import logger
from pathlib import Path
import torch
from scipy.stats import norm



URL = "http://localhost:7008/predict"


benchmark_data = np.genfromtxt("phrases/benchmark_8_128.tsv", delimiter="\t", dtype=str, skip_header=1)[:, -1]

payloads = [{"text": text} for text in benchmark_data]

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# payload = payloads[0]
for payload in payloads:
    logger.info(f"Sending request to {URL}")
    response = requests.post(URL, json=payload)

    if response.status_code == 200:
        logger.info("Request successful")

        response = response.json()
        encoded_audio = response["content"]
        decoded_audio = base64.b64decode(encoded_audio)

        if response["dtype"] == "float32":
            dtype = np.float32
        elif response["dtype"] == "float64":
            dtype = np.float64
        else:
            raise ValueError(f"Unsupported dtype: {response['dtype']}")

        audio = np.frombuffer(decoded_audio, dtype=dtype)
        write(output_dir/"test.wav", 22050, audio)
        logger.info(f"Audio file saved as {output_dir} / test.wav")

    else:
        logger.error(f"Request failed. {response.status_code}")



# Compute metrics
with open ("logs/tts_server.log", "r") as f:
    lines = f.readlines()


sampling_rate = 22050
inference_time = []
num_samples = []
num_utterances = []
for l in lines:
    info = l.strip('\n').split(',')[1:]

    info = [f.split(":") for f in info]
    info = {k: float(v) for k, v in info}

    inference_time.append(info["Inference_time"])
    num_samples.append(info["Samples"])
    num_utterances.append(info["Utterances"])

inference_time = np.sort(np.array(inference_time))
num_samples = np.array(num_samples)
num_utterances = np.array(num_utterances)

rtf = num_samples.sum() / (num_utterances.sum() * inference_time.mean() * sampling_rate)

logger.info(f"Average latency: {inference_time.mean():.4f}s")
logger.info(f"90%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.90) / 2) * inference_time.std():.4f}s")
logger.info(f"95%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.95) / 2) * inference_time.std():.4f}s")
logger.info(f"99%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.99) / 2) * inference_time.std():.4f}s")

logger.info(f"RTF: {rtf:.2f} x")

logger.info(f"Throughput: {int(num_samples.sum() / inference_time.sum()):,}")
