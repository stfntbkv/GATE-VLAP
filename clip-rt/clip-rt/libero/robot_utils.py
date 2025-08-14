"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

from clip_rt_utils import get_clip_rt
from clip_rt_utils import _get_clip_rt_action

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family == "clip_rt":
        model, _  = get_clip_rt()
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    resize_size = 224
    return resize_size


def get_clip_rt_action(
    model, preprocess, tokenizer, obs, task_label, device=DEVICE
):
    from PIL import Image

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    """Queries the model to get an action."""
    action = _get_clip_rt_action(
        model, preprocess, tokenizer, image, task_label, device
    )
    # assert action.shape == (ACTION_DIM,)
    return action

