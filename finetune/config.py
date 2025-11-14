#config.py
import torch, os
from transformers import BitsAndBytesConfig

# 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델/데이터 경로 및 이름
BASE_MODEL = "./DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME = "code_search_net"
DATASET_CONFIG = "python"   # "python", "java"
OUTPUT_DIR = "./deepseek-lora-output"

# 학습 하이퍼파라미터
BATCH_SIZE = 8
LR = 5e-5
NUM_EPOCHS = 3
SEED = 42

# LoRA 설정
LORA_PARAMS = {
    "r": 4,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
}

# 4-bit 양자화 (QLoRA)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Trainer 설정
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 1,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "logging_dir": "./logs",
    "fp16": True,
    "optim": "paged_adamw_8bit"
}