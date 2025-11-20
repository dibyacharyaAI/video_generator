import os
import requests

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("DIFFUSION_API_KEY")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "inputs": "Astronaut riding a horse",
    "options": {"wait_for_model": True}
}

resp = requests.post(API_URL, headers=headers, json=payload)
print("Status:", resp.status_code)
print("Content-Type:", resp.headers.get("content-type"))
print("Body:", resp.text[:1000])
