import glob
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional


def get_latest_checkpoint(dirpath) -> str:
    dirs = glob.glob(os.path.join(dirpath, "*step*"))
    assert len(dirs), f"no valid checkpoint directories found in {dirpath}"
    ckpt_dir = max(dirs, key=os.path.getmtime)
    return ckpt_dir

def timestamp() -> float:
    return datetime.now().timestamp()

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False
