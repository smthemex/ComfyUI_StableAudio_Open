import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model, create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import load_ckpt_state_dict, load_file
import json
import numpy as np
import random
import os
import sys
import folder_paths

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)
MAX_SEED = np.iinfo(np.int32).max

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "fma_dataset_attribution2.csv" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths != []:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


scheduler = ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"]


class Use_LocalModel_Or_Repo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repo_id",)
    FUNCTION = "repo_choice"
    CATEGORY = "StableAudio_Open"

    def repo_choice(self, local_model_path, repo_id):
        if repo_id == "":
            if local_model_path == "none":
                raise "you need fill repo_id or download model in diffusers directory "
            elif local_model_path != "none":
                model_path = get_local_path(file_path, local_model_path)
                repo_id = get_instance_path(model_path)
        elif repo_id != "none" and repo_id.find("\\") != -1:
            repo_id = get_instance_path(repo_id)
        return (repo_id,)


class StableAudio_Open:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True,"default": "A serenade suitable for assisting sleep"}),
                "step": ("INT", {"default": 100, "min": 10, "max": 40960, "step": 1, "display": "number"}),
                "cfg": ("INT", {"default": 7, "min": 1, "max": 50, "step": 1, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "display": "number"}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 4096.0, "step": 0.1, "display": "number"}),
                "sigma_max": ("FLOAT", {"default": 500, "min": 0.1, "max": 4096.0, "step": 0.1, "display": "number"}),
                "seconds_start": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "display": "number"}),
                "seconds_total": ("INT", {"default": 30, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "init_noise_level":("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "scheduler": (scheduler,)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "stableaudio_open"
    CATEGORY = "StableAudio_Open"

    def stableaudio_open(self, repo_id, prompt, step, cfg,batch_size, sigma_min, sigma_max, seconds_start, seconds_total,
                         init_noise_level, seed,scheduler):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if repo_id == "stabilityai/stable-audio-open-1.0":
            model, model_config = get_pretrained_model(repo_id)
        else:
            json_path = get_instance_path(os.path.join(repo_id, "model_config.json"))
            model_path = get_instance_path(os.path.join(repo_id, "model.safetensors"))
            with open(json_path) as f:
                model_config = json.load(f)
            model = create_model_from_config(model_config)
            model.load_state_dict(load_ckpt_state_dict(model_path), strict=False)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        model = model.to(device)
        model.to(torch.float16)

        #seed = np.random.randint(0, MAX_SEED)
        # Set up text and timing conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
        }]

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=step,
            cfg_scale=cfg,
            conditioning=conditioning,
            batch_size= batch_size,
            sample_size=sample_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampler_type=scheduler,
            device=device,
            seed=seed,
            init_noise_level=init_noise_level
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        filename_prefix = ''.join(random.choice("0123456789") for _ in range(5))
        output_path = os.path.join(file_path,"output",f"output_{filename_prefix}.wav")
        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(
            torch.int16).cpu()
        torchaudio.save(output_path, output, sample_rate)
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "Use_LocalModel_Or_Repo": Use_LocalModel_Or_Repo,
    "StableAudio_Open": StableAudio_Open
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Use_LocalModel_Or_Repo": "Use_LocalModel_Or_Repo",
    "StableAudio_Open": "StableAudio_Open"
}
