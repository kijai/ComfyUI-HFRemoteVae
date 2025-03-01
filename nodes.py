import torch
import json
import base64
import requests
from PIL import Image
from typing import cast, List, Literal, Optional, Union

from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from safetensors.torch import _tobytes

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}

def remote_decode(
    endpoint: str,
    tensor: torch.Tensor,
    processor: Optional[Union[VaeImageProcessor, VideoProcessor]] = None,
    do_scaling: bool = True,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["base64", "binary"] = "base64",
    output_tensor_type: Literal["base64", "binary"] = "base64",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image], bytes, torch.Tensor]:
    if tensor.ndim == 3 and height is None and width is None:
        raise ValueError("`height` and `width` required for packed latents.")
    if output_type == "pt" and partial_postprocess is True and processor is None:
        raise ValueError(
            "`processor` is required with `output_type='pt' and `partial_postprocess=False`."
        )
    headers = {}
    parameters = {
        "do_scaling": do_scaling,
        "output_type": output_type,
        "partial_postprocess": partial_postprocess,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    }
    if height is not None and width is not None:
        parameters["height"] = height
        parameters["width"] = width
    tensor_data = _tobytes(tensor, "tensor")
    if input_tensor_type == "base64":
        headers["Content-Type"] = "tensor/base64"
    elif input_tensor_type == "binary":
        headers["Content-Type"] = "tensor/binary"
    if output_type == "pil" and image_format == "jpg" and processor is None:
        headers["Accept"] = "image/jpeg"
    elif output_type == "pil" and image_format == "png" and processor is None:
        headers["Accept"] = "image/png"
    elif (output_tensor_type == "base64" and output_type == "pt") or (
        output_tensor_type == "base64"
        and output_type == "pil"
        and processor is not None
    ):
        headers["Accept"] = "tensor/base64"
    elif (output_tensor_type == "binary" and output_type == "pt") or (
        output_tensor_type == "binary"
        and output_type == "pil"
        and processor is not None
    ):
        headers["Accept"] = "tensor/binary"
    elif output_type == "mp4":
        headers["Accept"] = "text/plain"
    if input_tensor_type == "base64":
        kwargs = {"json": {"inputs": base64.b64encode(tensor_data).decode("utf-8")}}
    elif input_tensor_type == "binary":
        kwargs = {"data": tensor_data}
    response = requests.post(endpoint, params=parameters, **kwargs, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    if output_type == "pt" or (output_type == "pil" and processor is not None):
        if output_tensor_type == "base64":
            content = response.json()
            output_tensor = base64.b64decode(content["inputs"])
            parameters = content["parameters"]
            shape = parameters["shape"]
            dtype = parameters["dtype"]
        elif output_tensor_type == "binary":
            output_tensor = response.content
            parameters = response.headers
            shape = json.loads(parameters["shape"])
            dtype = parameters["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(
            bytearray(output_tensor), dtype=torch_dtype
        ).reshape(shape)
    if output_type == "pt":
        if partial_postprocess:
            output = [Image.fromarray(image.numpy()) for image in output_tensor]
            if len(output) == 1:
                output = output[0]
        else:
            if processor is None:
                output = output_tensor
            else:
                if isinstance(processor, VideoProcessor):
                    output = cast(
                        List[Image.Image],
                        processor.postprocess_video(output_tensor, output_type="pil")[0],
                    )
                else:
                    output = cast(
                        Image.Image,
                        processor.postprocess(output_tensor, output_type="pil")[0],
                    )
    elif output_type == "mp4":
        output = response.content
    return output

class RemoteVAE:
    def __init__(self, endpoint: str, vae_scale_factor: int = 8):
        self.endpoint = endpoint
        self.vae_scale_factor = vae_scale_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        result = remote_decode(
            endpoint=self.endpoint,
            tensor=latents,
            height=latents.shape[2] * self.vae_scale_factor,
            width=latents.shape[3] * self.vae_scale_factor,
            processor=None,
            output_type="pt",
            partial_postprocess=False,
            input_tensor_type="binary",
            output_tensor_type="binary",
            do_scaling=False
        )
        
        if "HunyuanVideo" in self.endpoint:
            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)
            video_processor.config.do_resize = False
            video = video_processor.postprocess_video(video=result, output_type="pt")
            out = video[0].permute(0, 2, 3, 1).cpu().float()
        else:
            image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            image_processor.config.do_resize = False
            result = image_processor.postprocess(result, output_type="pt")
            out = result.permute(0, 2, 3, 1).cpu().float()

        return out

class HFRemoteVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT",),
                    "VAE_type": (["Flux", "SDXL", "SD","HunyuanVideo"],),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "HFRemoteVae"

    def decode(self, samples, VAE_type):
        latents = samples["samples"]
        vae_scale_factor = 8

        if VAE_type == "HunyuanVideo":
            endpoint = "https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "Flux":
            endpoint = "https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "SDXL":
            endpoint = "https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "SD":
            endpoint = "https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/"

        result = remote_decode(
            endpoint=endpoint,
            tensor=latents,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            processor=None,
            output_type="pt",
            partial_postprocess=False,
            input_tensor_type="binary",
            output_tensor_type="binary",
            do_scaling=False
        )
        
        if VAE_type == "HunyuanVideo":
            video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
            video_processor.config.do_resize = False
            video = video_processor.postprocess_video(video=result, output_type="pt")
            out = video[0].permute(0, 2, 3, 1).cpu().float()
        else:
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            image_processor.config.do_resize = False
            result = image_processor.postprocess(result, output_type="pt")
            out = result.permute(0, 2, 3, 1).cpu().float()

        return (out,)

class HFRemoteVAE: # for nodes that require vae input. /decode only.
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "VAE_type": (["Flux", "SDXL", "SD","HunyuanVideo"],),
                    },
                }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "create_vae"
    CATEGORY = "HFRemoteVae"

    def create_vae(self, VAE_type):
        if VAE_type == "HunyuanVideo":
            endpoint = "https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "Flux":
            endpoint = "https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "SDXL":
            endpoint = "https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud/"
        elif VAE_type == "SD":
            endpoint = "https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/"

        vae = RemoteVAE(endpoint)
        return (vae,)

NODE_CLASS_MAPPINGS = {
    "HFRemoteVAEDecode": HFRemoteVAEDecode,
    "HFRemoteVAE": HFRemoteVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFRemoteVAEDecode": "HFRemoteVAEDecode",
    "HFRemoteVAE": "HFRemoteVAE",
}
