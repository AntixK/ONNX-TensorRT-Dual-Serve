import requests
import base64
import json
import numpy as np
from scipy.io.wavfile import write
from loguru import logger
from pathlib import Path
import torch
from scipy.stats import norm
from time import perf_counter
import aiohttp
import asyncio


URL = "http://localhost:7008/predict"


benchmark_data = np.genfromtxt("phrases/benchmark_8_128.tsv", delimiter="\t", dtype=str, skip_header=1)[:, -1]

payloads = [{"text": text} for text in benchmark_data]



output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


async def send_payload(payload: dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, data=payload) as response:
            print(f"Status: {response.status}")
            print(f"Response: {await response.text()}")

async def main():

    await send_payload(payloads[0])

if __name__ == "__main__":
    asyncio.run(main())



# inference_time = []
# # payload = payloads[0]
# for payload in payloads:
#     logger.info(f"Sending request to {URL}")
#     t1 = perf_counter()
#     response = requests.post(URL, json=payload)

#     t2 = perf_counter()
#     inference_time.append(t2 - t1)

#     if response.status_code == 200:
#         logger.info("Request successful")

#         response = response.json()
#         encoded_audio = response["content"]
#         decoded_audio = base64.b64decode(encoded_audio)

#         if response["dtype"] == "float32":
#             dtype = np.float32
#         elif response["dtype"] == "float64":
#             dtype = np.float64
#         else:
#             raise ValueError(f"Unsupported dtype: {response['dtype']}")

#         # audio = np.frombuffer(decoded_audio, dtype=dtype)
#         # write(output_dir/"test.wav", 22050, audio)
#         # logger.info(f"Audio file saved as {output_dir} / test.wav")

#     else:
#         logger.error(f"Request failed. {response.status_code}")

# client_inference_time =  np.sort(np.array(inference_time))
# logger.info(f"Average client latency: {client_inference_time.mean():.4f}s")
# logger.info(f"90%_client_latency: {client_inference_time.mean() + norm.ppf((1.0 + 0.90) / 2) * client_inference_time.std():.4f}s")
# logger.info(f"95%_client_latency: {client_inference_time.mean() + norm.ppf((1.0 + 0.95) / 2) * client_inference_time.std():.4f}s")
# logger.info(f"99%_client_latency: {client_inference_time.mean() + norm.ppf((1.0 + 0.99) / 2) * client_inference_time.std():.4f}s")


# # Compute metrics
# sampling_rate = 22050

# inference_time = np.sort(np.array(inference_time))
# num_utterances = np.array([1.0000]*len(benchmark_data))
# num_samples = np.array([169216.0]*len(benchmark_data))

# rtf = num_samples.sum() / (inference_time.mean() * sampling_rate)

# logger.info(f"Average latency: {inference_time.mean():.4f}s")
# logger.info(f"90%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.90) / 2) * inference_time.std():.4f}s")
# logger.info(f"95%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.95) / 2) * inference_time.std():.4f}s")
# logger.info(f"99%_server_latency: {inference_time.mean() + norm.ppf((1.0 + 0.99) / 2) * inference_time.std():.4f}s")

# logger.info(f"RTF: {rtf:.2f} x")

# logger.info(f"Throughput: {int(num_samples.sum() / inference_time.sum()):,}")
