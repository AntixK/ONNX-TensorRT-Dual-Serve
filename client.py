import requests
import base64
import numpy as np
from scipy.io.wavfile import write
from loguru import logger
from pathlib import Path

URL = "http://localhost:7008/predict"
payload = {"text": "I love this product"}
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

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
