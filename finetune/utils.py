# utils.py
import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, tokenizer, output_dir: str, epoch: int):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, f"model_epoch_{epoch}"))
    tokenizer.save_pretrained(os.path.join(output_dir, f"tokenizer_epoch_{epoch}"))

def log_config(config_module):
    # 설정값 출력
    print("=== Configuration ===")
    for attr in dir(config_module):
        if attr.isupper():
            print(f"{attr} = {getattr(config_module, attr)}")
    print("=====================")
