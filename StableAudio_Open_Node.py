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



def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


scheduler = ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"]
device = "cuda" if torch.cuda.is_available() else "cpu"

class StableAudio_ModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"}),
                "use_diffuser_pipe":("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("MODEL","DICT",)
    RETURN_NAMES = ("model","info",)
    FUNCTION = "loader_main"
    CATEGORY = "StableAudio_Open"

    def loader_main(self, local_model_path, repo_id,use_diffuser_pipe):
        if repo_id == "":
            if local_model_path == "none":
                raise "you need fill repo_id or download model in diffusers directory "
            elif local_model_path != "none":
                model_path = os.path.join(folder_paths.models_dir,"diffusers",local_model_path)
                repo_id = get_instance_path(model_path)
        else:
            repo_id = get_instance_path(repo_id)
        if use_diffuser_pipe:
            from diffusers import StableAudioPipeline
            model = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
            sample_rate=0
            sample_size=0
        else:
            if repo_id == "stabilityai/stable-audio-open-1.0":
                model, model_config = get_pretrained_model(repo_id)
            else:
                json_path = os.path.join(repo_id, "model_config.json")
                model_path = os.path.join(repo_id, "model.safetensors")
                with open(json_path) as f:
                    model_config = json.load(f)
                model = create_model_from_config(model_config)
                model.load_state_dict(load_ckpt_state_dict(model_path), strict=False)
            
            sample_rate = model_config["sample_rate"]
            sample_size = model_config["sample_size"]
           
            model.to(device)
            model.to(torch.float16)
        info = {"sample_rate": sample_rate, "sample_size": sample_size,"use_diffuser_pipe":use_diffuser_pipe }
        
        return (model,info)


class StableAudio_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "info":("DICT",),
                "prompt": ("STRING", {"multiline": True,"default": "The sound of a hammer hitting a wooden surface."}),
                "negative_prompt":("STRING", {"multiline": True,"default": "Low quality."}),
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

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "stableaudio_sampler"
    CATEGORY = "StableAudio_Open"

    def stableaudio_sampler(self, model,info, prompt,negative_prompt, step, cfg,batch_size, sigma_min, sigma_max, seconds_start, seconds_total,
                         init_noise_level, seed,scheduler):

        #seed = np.random.randint(0, MAX_SEED)
        # Set up text and timing conditioning
        sample_size=info["sample_size"]
        sample_rate=info["sample_rate"]
        use_diffuser_pipe=info["use_diffuser_pipe"]
        if use_diffuser_pipe:
            generator = torch.Generator("cuda").manual_seed(seed)
            end_in_s=float(seconds_total-seconds_start)
            audio = model(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=step,
                audio_end_in_s=end_in_s,
                num_waveforms_per_prompt=batch_size,
                generator=generator,
            ).audios
            waveform = audio[0].T.float().cpu().numpy()
            sample_rate=model.vae.sampling_rate
            #sf.write("hammer.wav", output, pipe.vae.sampling_rate)
        else:
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
                batch_size=batch_size,
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
            waveform = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(
                torch.int16).cpu()
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        #torchaudio.save(output_path, waveform, sample_rate)
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "StableAudio_ModelLoader": StableAudio_ModelLoader,
    "StableAudio_Sampler": StableAudio_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudio_ModelLoader": "StableAudio_ModelLoader",
    "StableAudio_Sampler": "StableAudio_Sampler"
}
