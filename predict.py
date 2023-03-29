import io
import os
import shutil
import subprocess
import urllib
import zipfile
import tarfile
import time
import torch

from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    load_pipeline_from_original_stable_diffusion_ckpt,
)


def dl_ckpt(model, version, path, cache_dir=None):
    return hf_hub_download(model, path, revision=version, cache_dir=cache_dir)


def load_ckpt(
    ckpt_path,
    upcast_attention=False,  # Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.
    extract_ema=False,  # Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"  or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.
    prediction_type=None,
    scheduler_type="pndm",
    pipeline_type=None,
    stable_unclip=None,
    from_safetensors=False,  # is in `safetensors` format, load checkpoint with safetensors instead of PyTorch
    stable_unclip_prior=None,
    controlnet=None,
    num_in_channels=None,
    clip_stats_path=None,
):
    if ckpt_path.endswith(".safetensors"):
        print("loading from safetensors")
        from_safetensors = True
    return load_pipeline_from_original_stable_diffusion_ckpt(
        checkpoint_path=ckpt_path,
        prediction_type=prediction_type,
        model_type=pipeline_type,
        extract_ema=extract_ema,
        scheduler_type=scheduler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        from_safetensors=from_safetensors,
        stable_unclip=stable_unclip,
        stable_unclip_prior=stable_unclip_prior,
        clip_stats_path=clip_stats_path,
        controlnet=controlnet,
    )


class Predictor(BasePredictor):
    def download_repo(self, repo_id=None, revision='main', ckpt_file=None, dest="weights", float16=True):
        """Download the model weights from the given URL"""
        print("Downloading weights...")

        local_ckpt_file = dl_ckpt(repo_id,  revision, ckpt_file, "workdir")
        pipe = load_ckpt(local_ckpt_file)
        shutil.rmtree("workdir")
        pipe.save_pretrained(dest)
        if float16:
            pipe = StableDiffusionPipeline.from_pretrained(dest, torch_dtype=torch.float16)
            shutil.rmtree(dest)
            pipe.save_pretrained(dest)

    def zip_dir(self, weights_dir, out_file):
        start = time.time()
        with zipfile.ZipFile(out_file, "w") as zip:
            directory = Path(weights_dir)
            print("adding to zip:")
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    print(file_path)
                    zip.write(file_path, arcname=file_path.relative_to(weights_dir))

        print("Made zip in {:.2f}s".format(time.time() - start))

    def predict(
        self,
        repo_id: str = Input(
            description="HF repo id: username/template",
        ),
        ckpt_file: str = Input(
            description="full path of ckpt file in repo",
        ),
        force_float16: bool = Input(
            description="ensure float16 for speed",
            default=True
        ),
        revision: str = Input(description="HF repo revision", default="main"),
    ) -> Path:
        weights_dir = "weights"
        if os.path.exists(weights_dir):
            shutil.rmtree(weights_dir)

        self.download_repo(repo_id=repo_id, revision=revision, ckpt_file=ckpt_file, dest=weights_dir, float16=force_float16)

        out_file = "output.zip"
        if os.path.exists(out_file):
            os.remove(out_file)

        self.zip_dir(weights_dir, out_file)

        return Path(out_file)
