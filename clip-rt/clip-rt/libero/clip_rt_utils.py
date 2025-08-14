"""Utils for evaluating the CLIP-RT policy."""

import json
import time
import torch
import numpy as np
from PIL import Image
import open_clip
from numpy.core.multiarray import scalar
from numpy import dtype
from numpy.dtypes import Float64DType

# Initialize important constants and pretty-printing mode in NumPy.
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Register safe globals for torch serialization
torch.serialization.add_safe_globals([scalar, dtype, Float64DType])

# Initialize system prompt for CLIP-RT
CLIP_RT_PROMPT = (
    "what motion should the robot arm perform to complete the instruction '{}'?"
)


def get_clip_rt(model_name="ViT-H-14-378-quickgelu", model_path="", task_split=""):
    """Loads and returns a CLIP-RT model from checkpoint."""
    print("[*] Instantiating Pretrained CLIP-RT model")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=model_path, use_action_decoder=True
    )
    model.eval()

    # Move model to device
    model = model.to(DEVICE)
    return model, preprocess


def get_tokenizer(model_name="ViT-H-14-378-quickgelu"):
    """Get CLIP-RT model's tokenizer."""
    return open_clip.get_tokenizer(model_name)


def _get_clip_rt_action(
    model,
    preprocess,
    tokenizer,
    image: Image.Image,
    task_label: str,
    device=DEVICE,
) -> list[list[float]]:
    """Generates an action with the CLIP-RT policy."""

    image = preprocess(image).unsqueeze(0).to(device)
    inst = tokenizer(CLIP_RT_PROMPT.format(task_label)).to(device)

    device_type = "mps" if device.type == "mps" else "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.amp.autocast(device_type=device_type):
        image_features = model.encode_image(image, normalize=True)
        text_features = model.encode_text(inst, normalize=True)

        dummy_tokens = torch.full((image_features.shape[0], 56), model.pad_id).to(
            device=image_features.device
        )
        out_features = model.decode_action(dummy_tokens, image_features, text_features)

        batch_size = out_features.shape[0]
        out_features = out_features[:, 2:, :]  # [32, 56, 1024]
        out_features = out_features.reshape(batch_size, model.num_action_chunk, -1)
        action = model.action_head(out_features)

        pred = action.squeeze(0).cpu().numpy().tolist()

    assert isinstance(pred, list)
    assert isinstance(pred[0], list)
    assert isinstance(pred[0][0], float)
    assert len(pred) == 8
    assert len(pred[0]) == 7

    return pred
