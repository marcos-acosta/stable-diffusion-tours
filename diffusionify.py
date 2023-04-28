from pathlib import Path
from PIL import Image
from tqdm import tqdm

import base64
import io
import numpy as np
import os
import requests


class Diffusioner():
    def __init__(self,
                 txt2img_endpoint,
                 prompt,
                 preprocessor_name,
                 model_name,
                 seed=-1,
                 n_steps=20,
                 sampler_name="Euler a",
                 batch_size=1,
                 weight=1.0):
        self.txt2img_endpoint = txt2img_endpoint
        self.prompt = prompt
        self.preprocessor_name = preprocessor_name
        self.model_name = model_name
        self.seed = seed
        self.n_steps = n_steps
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.weight = weight

    def diffusionify(self, image_arr: np.ndarray = None, image_path: str = None, output_dir: str = None):
        if not ((image_arr is not None) ^ bool(image_path and output_dir)):
            raise Exception("Either image_arr or both image_path and output_dir must be defined")
        if image_arr is not None:
            pil_input_image = Image.fromarray(image_arr)
            bytes_buffer = io.BytesIO()
            pil_input_image.save(bytes_buffer, format='PNG')
            image_string = base64.b64encode(bytes_buffer.getvalue()).decode('ascii')
        elif image_path:
            output_dir = Path(output_dir)
            output_path = output_dir / Path(image_path).name
            if os.path.exists(output_path):
                return
            with open(image_path, "rb") as f:
                image_string = base64.b64encode(f.read()).decode('ascii')

        controlnet_params = {
            "input_image": image_string,
            "module": self.preprocessor_name,
            "model": self.model_name,
            "enabled": True,
            "weight": self.weight,
        }

        params = {
            "prompt": self.prompt,
            "seed": self.seed,
            "steps": self.n_steps,
            "sampler_name": self.sampler_name,
            "batch_size": self.batch_size,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [controlnet_params]
                }
            }
        }

        response = requests.post(
            url=self.txt2img_endpoint,
            json=params
        )

        try:
            response_json = response.json()
        except Exception:
            raise Exception(f"Could not parse response to JSON. Response: {response}")
        
        if response.status_code != 200:
            raise Exception(f"Request returned a non-200 status. Response: {response}")
        
        output_image_b64 = response_json['images'][0]
        output_image = Image.open(io.BytesIO(base64.b64decode(output_image_b64.split(",", 1)[0])))

        if output_dir:
            output_image.save(str(output_path))
        else:
            return np.array(output_image)

    def diffusionify_dir(self, input_dir: str, output_dir: str):
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        sorted_filepaths = sorted(list(input_dir.iterdir()), 
                                  key=lambda filename: int(filename.stem.split("_")[1]))
        for input_image_path in tqdm(sorted_filepaths):
            self.diffusionify(image_arr=None, image_path=input_image_path, output_dir=output_dir)