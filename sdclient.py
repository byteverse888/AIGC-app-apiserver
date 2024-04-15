# https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
# python调用sdapi的例子：

import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

payload = {
    "prompt": "1girl, upper body, looking at viewer, overcoat, ..., (photorealistic:1.4), best quality, masterpiece",
    "negative_prompt":"EasyNegative, bad-hands-5, paintings, sketches, (worst quality:2), ..., NSFW, child, childish",
    "steps": 20,
    "sampler_name": "DPM++ SDE Karras",
    "width": 480,
    "height": 640,
    "restore_faces": True
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()

for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    PI = PngImagePlugin.PngInfo()
    PI.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=PI)


# python SDWEBAPI.py

